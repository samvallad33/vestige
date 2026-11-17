#!/usr/bin/env python3
"""
Deterministic, benchmark-agnostic builder for Issue #817:
Packs ConceptNet 5.7.0 English edges for {AtLocation, UsedFor, CapableOf, HasA}
into a ~480 KB sha256-pinned table for crates/vestige-core/data/conceptnet-5.7-4rel.tsv.

Usage:
    python scripts/build_conceptnet_table.py <path-to-conceptnet-assertions-5.7.0.csv.gz>
"""

import gzip
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

RELS = {
    "/r/AtLocation": "L",
    "/r/UsedFor": "U",
    "/r/CapableOf": "C",
    "/r/HasA": "H",
}
WORD_RE = re.compile(r"^[a-z]{3,20}$")
MAX_PER_SRC = 12
TARGET_BYTES = 480 * 1024


def extract_en_word(uri: str) -> str | None:
    if not uri.startswith("/c/en/"):
        return None
    rest = uri[6:]
    word = rest.split("/", 1)[0]
    if WORD_RE.match(word):
        return word
    return None


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: build_conceptnet_table.py <conceptnet-assertions-5.7.0.csv.gz>", file=sys.stderr)
        sys.exit(2)

    src_gz = Path(sys.argv[1])
    if not src_gz.exists():
        print(f"Missing input: {src_gz}", file=sys.stderr)
        sys.exit(1)

    h_in = hashlib.sha256()
    with src_gz.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h_in.update(chunk)
    print(f"Input sha256: {h_in.hexdigest()}")

    # Keep highest-weight edge per (src, dst, rel)
    best: dict[tuple[str, str, str], float] = {}
    with gzip.open(src_gz, "rt", encoding="utf-8", errors="replace") as gz:
        for line in gz:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            rel_code = RELS.get(parts[1])
            if rel_code is None:
                continue
            src = extract_en_word(parts[2])
            dst = extract_en_word(parts[3])
            if src is None or dst is None or src == dst:
                continue
            try:
                meta = json.loads(parts[4])
                w = float(meta.get("weight", 1.0))
            except Exception:
                w = 1.0
            if w < 1.0:
                continue
            key = (src, dst, rel_code)
            if w > best.get(key, 0.0):
                best[key] = w

    by_src: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
    for (src, dst, rel_code), w in best.items():
        by_src[src].append((w, rel_code, dst))

    # Deterministic global rank by weight desc, then (src, rel, dst) asc, capped per src
    candidates: list[tuple[float, str, str, str]] = []
    for src, edges in by_src.items():
        edges.sort(key=lambda e: (-e[0], e[1], e[2]))
        for w, rel_code, dst in edges[:MAX_PER_SRC]:
            candidates.append((w, src, rel_code, dst))

    candidates.sort(key=lambda e: (-e[0], e[1], e[2], e[3]))

    selected: list[tuple[str, str, str, int]] = []
    total_bytes = 0
    for w, src, rel_code, dst in candidates:
        w_q = min(65535, max(1, int(round(w * 100))))
        line = f"{src}\t{rel_code}\t{dst}\t{w_q}\n"
        b = len(line.encode("utf-8"))
        if total_bytes + b > TARGET_BYTES:
            break
        selected.append((src, rel_code, dst, w_q))
        total_bytes += b

    selected.sort(key=lambda e: (e[0], e[1], e[2]))
    out_path = Path(__file__).resolve().parent.parent / "crates" / "vestige-core" / "data" / "conceptnet-5.7-4rel.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{s}\t{r}\t{d}\t{w}\n" for s, r, d, w in selected).encode("utf-8")
    out_path.write_bytes(payload)

    out_sha = hashlib.sha256(payload).hexdigest()
    print(f"Wrote {len(selected)} edges ({len(payload)} bytes) to {out_path}")
    print(f"Output sha256: {out_sha}")


if __name__ == "__main__":
    main()
