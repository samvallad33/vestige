"""Prepare pinned LongRCA inputs offline. Never invokes a model or native store.

Example: python3 -m campaigns.longrca.prepare --upstream upstream/longrca \
    --output evidence/longrca-mini --split mini --limit 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .schema import canonical, digest, split_record

LOCK_PATH = Path(__file__).with_name("source.lock.json")
FIXTURE_PATH = Path(__file__).with_name("fixtures") / "trace.json"
EXPECTED = {"full": 1140, "mini": 200, "sample": 1}


def hash_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_status(upstream: Path, lock: dict[str, Any]) -> dict[str, Any]:
    missing, changed = [], []
    for relative, expected in lock["files"].items():
        path = upstream / relative
        if not path.is_file():
            missing.append(relative)
        elif hash_file(path) != expected:
            changed.append(relative)
    status = (
        "SOURCE_MISMATCH"
        if changed
        else "SOURCE_PENDING"
        if missing
        else "VERIFIED_PINNED_METADATA"
    )
    if status == "VERIFIED_PINNED_METADATA":
        metadata = json.loads((upstream / "dataset-info.json").read_text())
        if (
            metadata.get("sha") != lock["dataset_revision"]
            or metadata.get("id") != "CLoud5-real/longrca-bench"
        ):
            status = "SOURCE_MISMATCH"
            changed.append("dataset identity")
    return {
        "status": status,
        "missing": missing,
        "changed": changed,
        "dataset_revision": lock["dataset_revision"],
        "repository_commit": lock["repository_commit"],
        "source_lock_sha256": digest(lock),
        "paper": lock["paper"],
        "dataset": lock["dataset"],
        "dataset_license": lock["dataset_license"],
        "repository_license": lock["repository_license"],
        "license_gate": "local research preparation allowed; redistribution/public submission requires license review",
    }


def read_official(
    upstream: Path, split: str, lock: dict[str, Any]
) -> list[dict[str, Any]]:
    if split == "sample":
        return [json.loads((upstream / "official-sample.json").read_text())]
    name = "longrca-" + split + ".parquet"
    path = upstream / name
    if not path.is_file():
        raise FileNotFoundError(name)
    expected = lock["parquet"][name]
    if (
        path.stat().st_size != expected["bytes"]
        or hash_file(path) != expected["sha256"]
    ):
        raise ValueError("pinned parquet byte size or LFS SHA256 does not match")
    import pyarrow.parquet as pq

    records = pq.read_table(path).to_pylist()
    if len(records) != EXPECTED[split]:
        raise ValueError("official split cardinality mismatch")
    tree = json.loads((upstream / "dataset-tree.json").read_text()) + json.loads(
        (upstream / "dataset-tree-page2.json").read_text()
    )
    prefix = "data/" + split + "/"
    ids = {
        Path(row["path"]).stem
        for row in tree
        if row["type"] == "file" and row["path"].startswith(prefix)
    }
    if {row["question_ID"] for row in records} != ids or len(ids) != EXPECTED[split]:
        raise ValueError("parquet identities differ from the pinned official Git tree")
    return records


def _new_json(path: Path, value: Any, *, lines: bool = False) -> None:
    raw = (
        b"".join(canonical(item) + b"\n" for item in value)
        if lines
        else canonical(value) + b"\n"
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)


def prepare(
    upstream: Path,
    output: Path,
    *,
    split: str = "mini",
    limit: int | None = None,
    fixture: bool = False,
) -> dict[str, Any]:
    if split not in EXPECTED or (
        limit is not None and (type(limit) is not int or limit <= 0)
    ):
        raise ValueError("known split and a positive explicit limit required")
    # Exclusive destination prevents accidental replacement of frozen evidence.
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    lock = json.loads(LOCK_PATH.read_text())
    source = source_status(upstream, lock)
    records, blockers = [], []
    if fixture:
        records = [json.loads(FIXTURE_PATH.read_text())]
        state = "SYNTHETIC_FIXTURE_PREPARED"
    elif source["status"] != "VERIFIED_PINNED_METADATA":
        state = source["status"]
        blockers.append("stage unchanged source-lock metadata")
    else:
        try:
            records = read_official(upstream, split, lock)
            state = "OFFICIAL_DATA_PREPARED"
        except FileNotFoundError as exc:
            state = "DATA_PENDING"
            blockers.append("stage pinned file: " + str(exc))
        except ImportError:
            state = "DEPENDENCY_PENDING"
            blockers.append(
                "install local pyarrow to decode verified parquet; no provider dependency"
            )
        except ValueError as exc:
            state = "DATA_MISMATCH"
            blockers.append(str(exc))
    # Selection only uses stable source IDs; never select by gold role/root.
    selected = sorted(records, key=lambda row: row["question_ID"])
    ids = [row["question_ID"] for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate dataset IDs")
    if limit is not None:
        selected = selected[:limit]
    public, references, manifest = [], [], []
    # Validate ALL records before selection as well, exposing schema drift.
    for record in records:
        split_record(record)
    for record in selected:
        trace, reference = split_record(record)
        public.append(trace.as_dict())
        references.append(reference.as_dict())
        manifest.append(
            {
                "question_ID": trace.question_id,
                "source": trace.source,
                "steps": len(trace.history),
                "evidence_sha256": trace.evidence_sha256,
            }
        )
    (output / "scorer-only").mkdir(mode=0o700)
    _new_json(output / "public.jsonl", public, lines=True)
    _new_json(output / "scorer-only" / "references.jsonl", references, lines=True)
    _new_json(
        output / "manifest.json",
        {
            "schema": "longrca-manifest-v1",
            "kind": "synthetic_fixture" if fixture else "official",
            "split": split,
            "limit": limit,
            "selection_rule": "lexicographic question_ID, before gold access",
            "available_n": len(records),
            "selected_n": len(selected),
            "records": manifest,
            "public_sha256": hash_file(output / "public.jsonl"),
        },
    )
    _new_json(output / "source.json", source)
    readiness = {
        "status": state,
        "kind": "synthetic_fixture" if fixture else "official",
        "prepared_n": len(public),
        "available_n": len(records),
        "expected_split_n": EXPECTED[split],
        "inference_calls": 0,
        "provider_spend_usd": "0",
        "benchmark_scores": None,
        "source_status": source["status"],
        "blockers": blockers,
        "next_gates": [
            "independently qualify isolated native binding and receipt verification",
            "freeze same backbone, evidence, budgets and scored split",
            "obtain approval before any paid trajectory",
        ],
        "privacy": "mount only public.jsonl/manifest into solver; scorer-only must be inaccessible",
    }
    _new_json(output / "readiness.json", readiness)
    return readiness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=sorted(EXPECTED), default="mini")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="synthetic smoke only; never an official score",
    )
    args = parser.parse_args()
    result = prepare(
        args.upstream,
        args.output,
        split=args.split,
        limit=args.limit,
        fixture=args.fixture,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["prepared_n"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
