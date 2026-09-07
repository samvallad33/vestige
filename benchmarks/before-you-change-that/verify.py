#!/usr/bin/env python3
"""Verify retained public evidence. No model, server, or application is executed."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys


ARMS = ("control", "mcp-memory-service", "vestige")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_file(root, relative):
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts, "Unsafe manifest path")
    path = root / rel
    require(path.is_file() and not path.is_symlink(), "Missing or linked file: " + relative)
    ancestor = path.parent
    while ancestor != root:
        require(not ancestor.is_symlink(), "Linked ancestor")
        ancestor = ancestor.parent
    require(path.resolve().is_relative_to(root.resolve()), "Manifest file leaves bundle")
    return path


def verify_hashes(root, full=False):
    manifest = load(root / "PUBLIC-DERIVATION.json")
    selected = [x for x in manifest["files"] if full or x["included_in_git"]]
    require(bool(selected), "Empty evidence manifest")
    names = [x["path"] for x in selected]
    require(len(names) == len(set(names)), "Duplicate manifest paths")
    for item in selected:
        path = safe_file(root, item["path"])
        require(path.stat().st_size == item["bytes"] and file_sha(path) == item["public_sha256"],
                "Public evidence hash mismatch: " + item["path"])
    prefixes = {Path(name).parts[0] for name in names if len(Path(name).parts) > 1}
    actual = set()
    for prefix in prefixes:
        directory = root / prefix
        require(not directory.is_symlink(), "Linked evidence directory")
        for path in directory.rglob("*"):
            require(not path.is_symlink(), "Linked evidence entry")
            if path.is_file():
                actual.add(path.relative_to(root).as_posix())
            else:
                require(path.is_dir(), "Special evidence entry")
    expected = {name for name in names if len(Path(name).parts) > 1}
    require(actual == expected, "Evidence directories contain unlisted files")
    return len(selected)


def pae(payload_type, payload):
    kind = payload_type.encode("utf-8")
    return b"DSSEv1 " + str(len(kind)).encode() + b" " + kind + b" " + str(len(payload)).encode() + b" " + payload


def verify_signatures(evidence):
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as error:
        raise ValueError("Install requirements.txt to verify Ed25519, or request --hashes-only explicitly") from error
    expected = load(evidence / "vestige/signing-public.json")
    keys = load(evidence / "vestige/receipt-public-keys.json")["keys"]
    require(len({x["key_id"] for x in keys}) == len(keys), "Duplicate public key IDs")
    by_id = {x["key_id"]: x for x in keys}
    captured = load(evidence / "vestige/receipt-verification/receipts.json")
    require(len(captured) == 3, "This retained run must contain three captured retrieval envelopes")
    seen = set()
    rows = []
    for row in captured:
        require(row["response"]["isError"] is False, "Receipt fetch reported an error")
        value = row["response"]["value"]
        env = value["attestation"]["envelope"]
        require(len(env["signatures"]) == 1, "Expected one signature per captured envelope")
        sig = env["signatures"][0]
        require(sig["keyid"] == expected["key_id"], "Unexpected signing key")
        key = base64.b64decode(by_id[sig["keyid"]]["public_key_base64"], validate=True)
        require(hashlib.sha256(key).hexdigest() == expected["public_key_sha256"], "Public key fingerprint mismatch")
        payload = base64.b64decode(env["payload"], validate=True)
        signature = base64.b64decode(sig["sig"], validate=True)
        decoded = json.loads(payload)
        identifier = row["requested_receipt_id"]
        require(identifier == decoded["receiptId"] == value["receipt"]["receipt_id"], "Receipt ID mismatch")
        require(identifier not in seen, "Duplicate receipt ID")
        seen.add(identifier)
        public = Ed25519PublicKey.from_public_bytes(key)
        public.verify(signature, pae(env["payloadType"], payload))
        rejected = False
        try:
            public.verify(signature, pae(env["payloadType"], payload + b" "))
        except InvalidSignature:
            rejected = True
        require(rejected, "Modified signed payload was accepted")
        rows.append({"receipt_id": identifier, "signature_valid": True, "modified_payload_rejected": True,
                     "signed_payload_sha256": hashlib.sha256(payload).hexdigest()})
    return rows


def result_value(result):
    if "structuredContent" in result:
        return result["structuredContent"]
    for part in result.get("content", []):
        if part.get("type") == "text":
            try:
                return json.loads(part["text"])
            except (ValueError, KeyError):
                pass
    return result.get("value", result)


def verify_observations(evidence):
    result = load(evidence / "result.json")
    require([x["arm"] for x in result["reports"]] == list(ARMS), "Wrong comparison arms")
    summaries = []
    for arm in result["reports"]:
        evaluation = arm["evaluation"]
        require(len(evaluation["checks"]) == 21 and all(c["pass"] is True for c in evaluation["checks"]), "Recorded checks differ")
        require(evaluation["score"] == evaluation["total"] == 21, "Recorded score differs")
        model = arm["model_run"]
        expected_timeout = arm["arm"] == "mcp-memory-service"
        require(model["timeout"] is expected_timeout, "Retained interruption status changed")
        require(model["exit_code"] == 0 and model["stdout_closed"] is True, "Recorded process state differs")
        transcript = [json.loads(line) for line in (evidence / (arm["arm"] + ".jsonl")).read_text().splitlines() if line]
        seq = [row["seq"] for row in transcript]
        require(len(seq) == len(set(seq)) and seq == sorted(seq), "Transcript sequence is duplicated or reordered")
        if not expected_timeout:
            require(any(x.get("event", {}).get("type") == "turn.completed" for x in transcript), "Missing natural model completion")
        summaries.append({"arm": arm["arm"], "recorded_checks": "21/21", "model_finished_naturally": not expected_timeout,
                          "model_timeout": model["timeout"], "events": len(transcript)})
    lifecycle = load(evidence / "vestige/intention-lifecycle.json")
    identifier = lifecycle["intention_id"]
    require(lifecycle["stored_before"]["status"] == "active", "Intention was not active before work")
    require(lifecycle["stored_after"]["status"] == "fulfilled", "Intention completion is not retained")
    require(lifecycle["status"] == "COMPLETED_AFTER_APPLICATION_PASS" and lifecycle["application_passed"] is True,
            "Unexpected coordinator lifecycle outcome")
    calls = load(evidence / "vestige/native-transactions.json")
    found = []
    for call in calls:
        args = call.get("arguments", {})
        if call.get("tool") != "intention" or not call.get("success") or args.get("action") != "check":
            continue
        if args.get("context", {}).get("file") != "src/ledger/reconcile.rs":
            continue
        response = result_value(call["result"])
        found.extend(x for x in response.get("triggered", []) if x.get("id") == identifier and x.get("status") == "active")
    require(any(x["description"] == lifecycle["stored_before"]["content"] for x in found), "Exact model-session intention delivery missing")
    return {"arms": summaries, "exact_native_intention_delivery": True, "recorded_coordinator_completion": True,
            "three_way_natural_completion": False}


def verify(root, full=False, hashes_only=False):
    report = {"public_file_hashes_checked": verify_hashes(root, full)}
    if hashes_only:
        report.update(status="PASS", scope="File hashes only; no signature or application verification")
        return report
    report["observations"] = verify_observations(root / "evidence")
    report["signatures"] = verify_signatures(root / "evidence")
    report.update(status="PASS", scope="Public file integrity, retained outcomes, native intention delivery, and three retrieval signatures. No model rerun; no application execution; no causal or governance guarantee.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--full-evidence", action="store_true", help="Verify every file in the extracted release evidence archive")
    parser.add_argument("--hashes-only", action="store_true")
    parser.add_argument("--output", type=Path, help="Create a new JSON report; never overwrite")
    args = parser.parse_args()
    try:
        report = verify(args.root.resolve(), args.full_evidence, args.hashes_only)
        if args.output:
            with args.output.open("x") as stream:
                json.dump(report, stream, indent=2)
                stream.write("\n")
        print(json.dumps(report, indent=2))
    except Exception as error:
        print("FAIL: " + str(error), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
