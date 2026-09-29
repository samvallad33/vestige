#!/usr/bin/env python3
"""Compare one migrated STRATA dump against its v3 source store.

Exit 0 when mapped counts, FSRS/node columns, tombstones, and legacy-edge
marking all hold. Prints a one-line verdict plus details.
"""
import glob
import json
import os
import shutil
import sqlite3
import sys
import tempfile

VOCAB = {
    "touched",
    "anchored_to",
    "derived_from",
    "supersedes",
    "corrects",
    "closed_by",
    "projected_to",
    "evidence_of",
}
# Similarity / narrative / backfill links are legacy data. They must not
# become a walkable causal edge (vocabulary type with legacy_inferred false).
SIMILARITY = {
    "semantic",
    "similar",
    "similarity",
    "narrative",
    "related",
    "backfill_candidate",
    "associative",
}
MAPPED = {
    "knowledge_nodes",
    "memory_connections",
    "fsrs_cards",
    "sync_tombstones",
    "deletion_tombstones",
}
FSRS_COLS = [
    "stability",
    "difficulty",
    "reps",
    "lapses",
    "next_review",
    "retention_strength",
    "storage_strength",
    "retrieval_strength",
    "valid_from",
    "valid_until",
    "scope",
    "suppression_count",
    "suppressed_at",
    "protected",
    "content_hash",
]


def source_snapshot(dbdir):
    tmp = tempfile.mkdtemp(prefix="v3src-")
    for f in glob.glob(os.path.join(dbdir, "vestige.db*")):
        shutil.copy(f, tmp)
    db = os.path.join(tmp, "vestige.db")
    # Normal open applies a copied WAL. The original directory is not opened.
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    schema = c.execute("select max(version) from schema_version").fetchone()[0]
    tables = [
        r[0]
        for r in c.execute(
            "select name from sqlite_master where type='table' "
            "and name not like 'sqlite_%' and name not like 'knowledge_fts%'"
        )
    ]
    counts = {}
    for name in tables:
        try:
            counts[name] = c.execute(f'select count(*) from "{name}"').fetchone()[0]
        except sqlite3.Error as e:
            counts[name] = f"ERR {e}"
    node_cols = [r[1] for r in c.execute("pragma table_info(knowledge_nodes)")]
    nodes = []
    for row in c.execute("select * from knowledge_nodes"):
        nodes.append({k: row[k] for k in row.keys()})
    edges = []
    if "memory_connections" in counts:
        for row in c.execute("select * from memory_connections"):
            edges.append({k: row[k] for k in row.keys()})
    tombs = {"sync_tombstones": [], "deletion_tombstones": []}
    if "sync_tombstones" in tables:
        for row in c.execute("select * from sync_tombstones"):
            tombs["sync_tombstones"].append(row["row_id"])
    if "deletion_tombstones" in tables:
        for row in c.execute("select * from deletion_tombstones"):
            tombs["deletion_tombstones"].append(row["memory_id"])
    walk = None
    if "walk_receipts" in tables:
        walk = c.execute("select count(*) from walk_receipts").fetchone()[0]
    c.close()
    shutil.rmtree(tmp)
    return {
        "schema": schema,
        "counts": counts,
        "node_cols": node_cols,
        "nodes": nodes,
        "edges": edges,
        "tombs": tombs,
        "walk_receipts": walk,
    }


def close_enough(src, legacy):
    if src is None:
        return legacy in ("", None)
    if isinstance(src, float):
        try:
            return abs(float(legacy) - src) <= 1e-9 * max(1.0, abs(src))
        except (TypeError, ValueError):
            return False
    if isinstance(src, int) and not isinstance(src, bool):
        return str(src) == str(legacy)
    return str(src) == str(legacy)


def main():
    dbdir, dump_path, skipped_path = sys.argv[1:4]
    src = source_snapshot(dbdir)
    dump = json.load(open(dump_path))
    skipped = []
    if os.path.exists(skipped_path):
        skipped = [s for s in open(skipped_path).read().split("\n") if s.strip()]
    problems = []
    kind = dump["kind_counts"]
    nodes = dump["nodes"]
    edges = dump["edges"]
    tombs = dump["tombs"] if "tombs" in dump else dump["tombstones"]
    receipt = dump.get("receipt") or {}
    receipt_counts = {k: v for k, v in receipt.get("counts") or []}

    src_nodes = src["counts"].get("knowledge_nodes", 0)
    src_edges = src["counts"].get("memory_connections", 0)
    src_sync = src["counts"].get("sync_tombstones", 0)
    src_del = src["counts"].get("deletion_tombstones", 0)
    walk = src["walk_receipts"]
    expect_nodes = src_nodes + (walk or 0)
    if kind.get("NODE", 0) != expect_nodes:
        problems.append(f"NODE frames {kind.get('NODE', 0)} != knowledge_nodes {src_nodes} + walk_receipts {walk}")
    if len(nodes) != expect_nodes:
        problems.append(f"decoded nodes {len(nodes)} != {expect_nodes}")
    if kind.get("EDGE", 0) != src_edges or len(edges) != src_edges:
        problems.append(f"EDGE {kind.get('EDGE', 0)} decoded {len(edges)} != memory_connections {src_edges}")
    if kind.get("TOMBSTONE", 0) != src_sync + src_del:
        problems.append(
            f"TOMBSTONE {kind.get('TOMBSTONE', 0)} != sync {src_sync} + deletion {src_del}"
        )

    by_id = {n["legacy_id"]: n for n in nodes}
    wanted = [c for c in FSRS_COLS if c in src["node_cols"]]
    wanted += [c for c in src["node_cols"] if c.startswith("source_")]
    if "author_actor_did" in src["node_cols"]:
        wanted.append("author_actor_did")
    # unique preserve order
    seen = set()
    wanted = [c for c in wanted if not (c in seen or seen.add(c))]
    missing_col_nodes = 0
    value_mismatch = 0
    examples = []
    for row in src["nodes"]:
        node = by_id.get(row["id"])
        if node is None:
            problems.append(f"missing node {row['id']}")
            continue
        if node["content"] != row["content"]:
            problems.append(f"content mismatch {row['id']}")
        if node["node_type"] != row["node_type"]:
            problems.append(f"type mismatch {row['id']}")
        legacy = dict(node["legacy"])
        for col in wanted:
            key = f"knowledge_nodes.{col}"
            if key not in legacy:
                missing_col_nodes += 1
                if len(examples) < 6:
                    examples.append(f"missing {key} on {row['id']}")
                continue
            if not close_enough(row[col], legacy[key]):
                value_mismatch += 1
                if len(examples) < 8:
                    examples.append(
                        f"{key} src={row[col]!r} legacy={legacy[key]!r} id={row['id'][:8]}"
                    )
    if missing_col_nodes:
        problems.append(f"missing carried columns: {missing_col_nodes}")
    if value_mismatch:
        problems.append(f"column value mismatches: {value_mismatch}")

    src_edges_by = {}
    for e in src["edges"]:
        src_edges_by.setdefault((e["source_id"], e["target_id"], e["link_type"]), []).append(e)
    causal_promotions = []
    for e in edges:
        lt = e["legacy_link_type"]
        if lt not in VOCAB:
            # Non-vocabulary links (semantic, narrative, backfill_candidate, …)
            # must stay legacy data: derived_from + legacy_inferred, not a
            # walkable causal edge.
            if e["link_type"] != "derived_from" or not e["legacy_inferred"]:
                causal_promotions.append(
                    {
                        "legacy_link_type": lt,
                        "link_type": e["link_type"],
                        "legacy_inferred": e["legacy_inferred"],
                    }
                )
        elif lt in VOCAB and e["legacy_inferred"]:
            problems.append(f"real vocab edge marked inferred: {lt}")
        elif lt in VOCAB and e["link_type"] != lt:
            problems.append(f"vocab edge rewritten: {lt} -> {e['link_type']}")
    # de-dup promotions
    if causal_promotions:
        problems.append(
            "similarity/legacy links became walkable causal edges: "
            + json.dumps(causal_promotions[:6])
        )

    tomb_ids = {(t["origin_table"], t["row_id"]) for t in tombs}
    for origin, ids in src["tombs"].items():
        for rid in ids:
            if (origin, rid) not in tomb_ids:
                problems.append(f"missing tombstone {origin} {rid}")

    nonempty = {k: v for k, v in src["counts"].items() if isinstance(v, int) and v > 0}
    skipped_set = set(skipped)
    unaccounted = []
    for name, n in sorted(nonempty.items()):
        if name == "node_embeddings":
            if receipt.get("dropped_vectors") != n:
                problems.append(
                    f"dropped_vectors {receipt.get('dropped_vectors')} != node_embeddings {n}"
                )
            continue
        if name in MAPPED or name == "walk_receipts":
            rc = receipt_counts.get(name)
            if name == "walk_receipts" and (walk or 0) == 0:
                continue
            if n == 0:
                continue
            if rc != n:
                problems.append(f"receipt count {name}={rc} source={n}")
            continue
        if name not in skipped_set:
            unaccounted.append(f"{name}={n}")
    if unaccounted:
        problems.append("nonempty tables neither mapped nor skipped: " + ", ".join(unaccounted))

    if walk is None:
        # table absent: must not have failed the migration; no walk nodes required
        pass
    elif walk == 0:
        pass
    else:
        walk_nodes = [n for n in nodes if n["node_type"] == "walk_receipt"]
        if len(walk_nodes) != walk:
            problems.append(f"walk_receipt nodes {len(walk_nodes)} != {walk}")

    if not receipt.get("signature_ok"):
        problems.append("receipt signature_ok is false")
    if not receipt.get("checksum_ok"):
        problems.append("receipt checksum_ok is false")
    if dump.get("decode_errors"):
        problems.append("decode errors: " + "; ".join(dump["decode_errors"][:4]))
    key = dump.get("strata_key") or {}
    if key.get("equals_old_source_hash_derivation") is True:
        problems.append("strata.key equals the old source-hash derivation")

    print(json.dumps({
        "schema": src["schema"],
        "source_nodes": src_nodes,
        "source_edges": src_edges,
        "source_link_types": sorted({e["link_type"] for e in src["edges"]}),
        "walk_receipts_table": walk,
        "kind_counts": kind,
        "receipt_counts": receipt_counts,
        "skipped": skipped,
        "fsrs_columns_checked": wanted,
        "suppressed_nodes": [
            r["id"] for r in src["nodes"] if (r.get("suppression_count") or 0) or r.get("protected")
        ],
        "problems": problems,
        "examples": examples,
        "edges_out": [
            {
                "legacy_link_type": e["legacy_link_type"],
                "link_type": e["link_type"],
                "legacy_inferred": e["legacy_inferred"],
            }
            for e in edges
        ],
    }, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
