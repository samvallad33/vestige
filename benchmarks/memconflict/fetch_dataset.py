#!/usr/bin/env python3
"""Fetch the pinned MemConflict dataset revision and verify its hash.

The dataset is NOT vendored into this repository. It is downloaded from the
upstream GitHub repo at the exact commit pinned in DATASET.lock.json and
verified by SHA-256 before use. If the hash does not match, this script fails
loudly rather than benchmarking against unknown data.

Usage:
    python3 fetch_dataset.py                 # fetch + verify into ./data/
    python3 fetch_dataset.py --verify-only   # verify an existing copy
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
LOCK = HERE / "DATASET.lock.json"
DATA_DIR = HERE / "data"
RAW = "https://raw.githubusercontent.com/TaoZhen1110/MemConflict/{rev}/{path}"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_lock() -> dict:
    return json.loads(LOCK.read_text())


def local_path_for(rel: str) -> pathlib.Path:
    return DATA_DIR / pathlib.Path(rel).name


def fetch(verify_only: bool = False) -> pathlib.Path:
    lock = load_lock()
    rev = lock["revision"]
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    primary = None
    for rel, meta in lock["files"].items():
        dest = local_path_for(rel)
        if not dest.exists():
            if verify_only:
                sys.exit(f"FAIL: {dest} missing (run without --verify-only to download)")
            url = RAW.format(rev=rev, path=rel)
            print(f"downloading {url}\n         -> {dest}", flush=True)
            urllib.request.urlretrieve(url, dest)

        actual = sha256_file(dest)
        expected = meta["sha256"]
        if actual != expected:
            sys.exit(
                f"FAIL: hash mismatch for {dest}\n"
                f"  expected {expected}\n"
                f"  actual   {actual}\n"
                "Refusing to benchmark against unverified data."
            )
        size = dest.stat().st_size
        if size != meta["bytes"]:
            sys.exit(f"FAIL: size mismatch for {dest}: {size} != {meta['bytes']}")
        print(f"OK  {dest.name}  sha256={actual[:16]}...  bytes={size}")
        if primary is None:
            primary = dest

    print(f"\ndataset pinned at revision {rev}")
    return primary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    fetch(verify_only=args.verify_only)
