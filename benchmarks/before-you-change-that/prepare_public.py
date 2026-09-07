#!/usr/bin/env python3
"""Create a public derivative of the retained local archive, without model calls.

The private archive is read-only. External links, native databases, signing
seeds, local agent state, and compiled programs are not publication artifacts.
Every included file receives a source hash, public hash, and transformation note.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from datetime import datetime, timezone


RUN = "context-intention-1ux5s5ji"
ARMS = ("control", "mcp-memory-service", "vestige")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def public_text(text, aliases=()):
    # These are presentation aliases, never executable host paths.
    for original, replacement in aliases:
        text = text.replace(original, replacement)
    text = re.sub(r"/Users/[^/\s\"'<>]+", "@recorded-home@", text)
    text = re.sub(r"/home/[^/\s\"'<>]+", "@recorded-home@", text)
    text = text.replace("@recorded-home@/.codex", "@recorded-agent-runtime@")
    text = text.replace("@recorded-home@/.claude", "@recorded-agent-runtime@")
    text = text.replace("@recorded-home@/.agents", "@recorded-agent-resources@")
    text = text.replace("@recorded-home@/.cache/codex-runtimes", "@recorded-agent-dependencies@")
    text = text.replace("@recorded-home@/.kimi-code", "@recorded-agent-runtime@")
    text = text.replace("@recorded-home@/Movies/Vestige", "@recorded-media@")
    text = text.replace("/private/tmp/", "@recorded-temp@/")
    text = text.replace("/private/var/folders/", "@recorded-temp@/")
    return text


def audit_public_files(root):
    """Reject residual private paths or secret material before publication."""
    forbidden = re.compile(
        r"/(?:Users|home)/[^/\s\"'<>]+|@recorded-home@/\.(?:codex|claude|agents|kimi-code)"
        r"|-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"
        r"|\b(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_]{20,}"
        r"|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|\bxox[baprs]-[A-Za-z0-9-]{20,}"
        r"|\b(?:sk|rk)_live_[A-Za-z0-9]{20,}|\bAIza[A-Za-z0-9_-]{35}"
    )
    operational_id = re.compile(r'"(?:thread_id|threadId|agent_id|agentId|turn_id|turnId)"\s*:\s*"[0-9a-f-]{36}"')
    for path in root.rglob("*"):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ValueError("Unsafe public file type: " + str(path.relative_to(root)))
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            if forbidden.search(text):
                raise ValueError("Private path or credential in public file: " + str(path.relative_to(root)))
            for _ in range(5):
                if operational_id.search(text):
                    raise ValueError("Unaliased agent operation ID: " + str(path.relative_to(root)))
                unescaped = text.replace('\\"', '"')
                if unescaped == text:
                    break
                text = unescaped


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def jbytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def compact_evidence(rel):
    parts = rel.parts
    if len(parts) == 1:
        return rel.suffix in (".json", ".jsonl")
    if parts[0] == "inputs":
        return True
    if parts[0] in ARMS:
        if len(parts) == 2:
            return rel.suffix in (".json", ".jsonl")
        return parts[1] in ("project", "receipt-verification", "coordinator-intention-wire")
    return False


def export(source, destination, full):
    run = source / "Demo" / "evidence" / RUN
    evidence = destination / "evidence"
    if evidence.exists() or full.exists():
        raise ValueError("Output evidence directories must not already exist")
    full.mkdir(parents=True)
    included, excluded = [], []
    archive = json.loads((source / "ARCHIVE-MANIFEST.json").read_text())
    original_manifest = next(x for x in archive["files"] if x["path"] == "Demo/evidence/" + RUN + "/manifest.json")
    original_run = Path(original_manifest["source"]).parent
    original_harness = original_run.parent.parent
    original_workspace = original_harness.parent.parent
    aliases = [(str(original_run), "@recorded-run@"),
               (str(original_harness), "@recorded-harness@"),
               (str(original_harness.parent), "@recorded-shared-harness@"),
               (str(original_workspace), "@recorded-workspace@")]
    for item in archive["files"]:
        if item["path"].startswith("Recorded runtime/") and item.get("source"):
            aliases.append((item["source"], "@recorded-runtime@/" + Path(item["path"]).name))
            runtime_path = Path(item["source"])
            if runtime_path.is_relative_to(original_workspace):
                aliases.append((str(runtime_path.relative_to(original_workspace)),
                                "@recorded-runtime@/" + Path(item["path"]).name))
    # Local agent/session IDs are not memory-record IDs. Keep stable public aliases.
    thread_ids = set()
    for path in run.rglob("*.json*"):
        text = path.read_text()
        for _ in range(5):
            thread_ids.update(re.findall(r'"(?:thread_id|threadId|agent_id|agentId|turn_id|turnId)"\s*:\s*"([0-9a-f-]{36})"', text))
            unescaped = text.replace('\\"', '"')
            if unescaped == text:
                break
            text = unescaped
    recorded = json.loads((run / "result.json").read_text())
    for arm in recorded["reports"]:
        thread_ids.update(arm["model_run"].get("thread_ids", []))
    aliases.extend((identifier, "recorded-agent-" + sha(identifier.encode())[:12]) for identifier in sorted(thread_ids))
    aliases.sort(key=lambda pair: len(pair[0]), reverse=True)
    roots = [(run, Path("evidence")),
             (source / "Demo", Path("historical-harness")),
             (source / "Shared harness", Path("historical-shared-harness")),
             (source / "Supporting evidence", Path("historical-campaign-checks"))]
    for root, prefix in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            rel = path.relative_to(root)
            if prefix == Path("historical-harness") and any(
                    part in ("evidence", "runtime", "__pycache__", ".pytest_cache") for part in rel.parts):
                continue
            original = path.read_bytes()
            target = prefix / rel
            if path.suffix in (".seed", ".db", ".sqlite", ".pyc") or "receipt-keys" in rel.parts:
                excluded.append({"path": str(target), "source_sha256": sha(original),
                                 "reason": "Private key, native database, or runtime state; not public evidence."})
                continue
            try:
                text = original.decode("utf-8")
                if "\x00" in text:
                    raise ValueError("binary")
            except (UnicodeDecodeError, ValueError):
                excluded.append({"path": str(target), "source_sha256": sha(original),
                                 "reason": "Compiled or binary artifact; text inputs and observations retained."})
                continue
            public = public_text(text, aliases).encode()
            write(full / target, public)
            (full / target).chmod(path.stat().st_mode & 0o777)
            committed = prefix == Path("evidence") and compact_evidence(rel)
            if committed:
                write(destination / target, public)
                (destination / target).chmod(path.stat().st_mode & 0o777)
            included.append({"path": str(target), "source_sha256": sha(original),
                             "public_sha256": sha(public), "bytes": len(public),
                             "transformation": "none" if public == original else "workstation-role-and-agent-id-aliases",
                             "included_in_git": committed})

    # Export only public verification keys; never export seed material.
    keys = []
    uri = (run / "vestige/server/data/vestige.db").resolve().as_uri() + "?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as db:
        for key_id, public_key in db.execute("SELECT key_id, public_key FROM receipt_signing_keys ORDER BY key_id"):
            keys.append({"key_id": key_id, "public_key_base64": base64.b64encode(public_key).decode(),
                         "public_key_sha256": sha(public_key)})
    key_data = jbytes({"keys": keys, "trust_boundary": "Keys from the retained disposable demo store; not external identity certification."})
    key_rel = Path("evidence/vestige/receipt-public-keys.json")
    write(destination / key_rel, key_data)
    write(full / key_rel, key_data)
    included.append({"path": str(key_rel), "source_sha256": None, "public_sha256": sha(key_data),
                     "bytes": len(key_data), "transformation": "public-key-only-export", "included_in_git": True})

    capture_bytes = (source / "Video/capture-start.json").read_bytes()
    capture = json.loads(capture_bytes)
    def utc(milliseconds):
        return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    timing = {"schema": "vestige.recorded-video-timing.v1",
              "capture_context_utc": utc(capture["captureContextAt"]),
              "dispatch_utc": utc(capture["dispatchAt"]),
              "replay_lead_seconds": (capture["dispatchAt"] - capture["captureContextAt"]) / 1000,
              "pages": [{"view": page["view"], "page_opened_utc": utc(page["pageOpenedAt"])} for page in capture["pages"]],
              "source_capture_metadata_sha256": sha(capture_bytes),
              "boundary": "Derived recording timing only. Public videos replay retained events; they are not the original screen recordings."}
    timing_data = jbytes(timing)
    timing_rel = Path("evidence/recording-timing.json")
    write(destination / timing_rel, timing_data)
    write(full / timing_rel, timing_data)
    included.append({"path": str(timing_rel), "source_sha256": sha(capture_bytes), "public_sha256": sha(timing_data),
                     "bytes": len(timing_data), "transformation": "recorded-capture-timing-export", "included_in_git": True})

    # Signed DSSE envelope values must remain exactly those captured in the run.
    rel = Path("vestige/receipt-verification/receipts.json")
    original_receipts = json.loads((run / rel).read_text())
    public_receipts = json.loads((evidence / rel).read_text())
    envelopes = []
    for original, public in zip(original_receipts, public_receipts, strict=True):
        a = original["response"]["value"]["attestation"]["envelope"]
        b = public["response"]["value"]["attestation"]["envelope"]
        if a != b:
            raise ValueError("Publication changed a signed retrieval envelope")
        decoded = base64.b64decode(a["payload"], validate=True)
        if public_text(decoded.decode(), aliases) != decoded.decode():
            raise ValueError("Signed payload contains a private path; cannot publish this envelope unchanged")
        envelopes.append({"receipt_id": original["requested_receipt_id"],
                          "signed_payload_sha256": sha(decoded), "envelope_values_unchanged": True})
    manifest = {"schema": "vestige.public-demo-derivative.v1", "source_run": RUN,
                "source_archive_manifest_sha256": sha((source / "ARCHIVE-MANIFEST.json").read_bytes()),
                "files": included, "excluded_files": excluded, "signed_envelopes": envelopes,
                "other_exclusions": ["Local agent runtime/session state", "Account/external symlink metadata",
                                     "Private signing seeds", "Native databases and installed runtime binaries",
                                     "Original video frames with private paths; supplied separately as labeled event replays"],
                "boundary": "Public derivatives have their own hashes. Original local seals are retained as historical evidence, not asserted to validate redacted bytes."}
    for root in (destination, full):
        write(root / "PUBLIC-DERIVATION.json", jbytes(manifest))
    audit_public_files(full)
    audit_public_files(evidence)
    print(json.dumps({"public_text_files": len(included), "git_evidence_files": sum(x["included_in_git"] for x in included),
                      "excluded_binary_or_private_files": len(excluded), "unchanged_signed_envelopes": len(envelopes)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--full-evidence", required=True, type=Path)
    args = parser.parse_args()
    export(args.source.resolve(), args.destination.resolve(), args.full_evidence.resolve())


if __name__ == "__main__":
    main()
