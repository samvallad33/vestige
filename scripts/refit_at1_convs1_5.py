#!/usr/bin/env python3
"""
Step 4: Strictly Disjoint Convs 1-5 64-Tree Depth-4 Integer LambdaMART Refit for AT1.

STRICT DATA HYGIENE INVARIANT:
- Reads ONLY `benchmarks/data/locomo10.json[:5]` (Convs 1-5).
- NEVER reads, inspects, dumps, or fits on Convs 6-10 (`locomo10.json[5:]`).
- Validates generalization via 5-fold Leave-One-Conversation-Out CV across Convs 1-5.
- Outputs exact integer `AT1_TREE_ROOTS`, `AT1_TREE_NODES`, and SHA-256 `AT1_GBDT_SHA256`
  matching `crates/vestige-core/src/retrieval/seating/at1_scorer_q16.rs::canonical_gbdt_bytes()`.
"""
import argparse
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "benchmarks", "data", "locomo10.json")
AT1_RS = os.path.join(
    ROOT, "crates", "vestige-core", "src", "retrieval", "seating", "at1_scorer_q16.rs"
)
DEFAULT_BIN = r"C:\cargo-target\bench-fast\retrieval_bench.exe"
DEFAULT_DUMP = os.path.join(ROOT, "target_at1_convs1_5_dump.jsonl")

CAT_NAMES = {
    1: "Multi-Hop",
    2: "Temporal",
    3: "Open-Domain",
    4: "Single-Hop",
    5: "Adversarial",
}


def parse_evidence(ev_list):
    gold_turns = set()
    gold_sess = set()
    for raw in ev_list:
        for part in str(raw).split(";"):
            p = part.strip()
            if not p:
                continue
            gold_turns.add(p)
            if ":" in p:
                sess = p.split(":")[0]
                if sess.startswith("D"):
                    gold_sess.add(sess)
    return gold_sess, gold_turns


def dump_convs_1_5_features(bin_path, dump_path):
    if os.path.exists(dump_path):
        os.remove(dump_path)
    data = json.load(open(DATA_PATH, "r", encoding="utf-8"))
    # STRICTLY CONVS 1-5 ONLY — NEVER TOUCH CONVS 6-10!
    convs_1_5 = data[:5]

    env = os.environ.copy()
    env["VESTIGE_AT1_FEAT_DUMP"] = dump_path
    env.pop("VESTIGE_WATERFALL_LOG", None)

    proc = subprocess.Popen(
        [bin_path, "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=env,
    )

    qa_meta = []
    for ci, conv in enumerate(convs_1_5):
        conv_obj = conv["conversation"]
        sessions = []
        s_idx = 1
        while f"session_{s_idx}" in conv_obj:
            sid = f"D{s_idx}"
            sdate = conv_obj.get(f"session_{s_idx}_date_time", "")
            turns = []
            for t in conv_obj[f"session_{s_idx}"]:
                if not t.get("text") or not t.get("dia_id"):
                    continue
                txt = str(t.get("text", ""))
                cap = t.get("blip_caption", "")
                if cap:
                    txt = f"{txt}\n[caption] {cap}"
                turns.append(
                    {
                        "id": str(t["dia_id"]),
                        "role": str(t.get("speaker", "")),
                        "content": txt,
                    }
                )
            sessions.append({"id": sid, "date": sdate, "timestamp": 0, "turns": turns})
            s_idx += 1

        queries = []
        for qa in conv["qa"]:
            gs, gt = parse_evidence(qa.get("evidence", []))
            if not gs and not gt:
                continue
            queries.append({"text": qa["question"], "date": ""})
            qa_meta.append(
                {
                    "conv": ci,
                    "n_sessions": len(sessions),
                    "cat": int(qa["category"]),
                    "q": qa["question"],
                    "gold_s": gs,
                    "gold_t": gt,
                }
            )

        req = {"sessions": sessions, "queries": queries, "top_k": 10}
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        _resp = proc.stdout.readline()

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read()
        raise RuntimeError(f"retrieval_bench --worker failed: {err}")

    rows = [json.loads(l) for l in open(dump_path, "r", encoding="utf-8") if l.strip()]
    return align_dump_to_meta(qa_meta, rows)


def align_dump_to_meta(qa_meta, rows):
    """Align dumped AT1 rows (which skip queries that short-circuit before AT1) to qa_meta in order."""
    aligned_meta = []
    aligned_rows = []
    mi = 0
    for r in rows:
        rq = r["query"]
        rns = r["n_sessions"]
        while mi < len(qa_meta) and (qa_meta[mi]["q"] != rq or qa_meta[mi]["n_sessions"] != rns):
            mi += 1
        assert mi < len(qa_meta), f"Failed to align dump row query={rq!r} n_sessions={rns}"
        aligned_meta.append(qa_meta[mi])
        aligned_rows.append(r)
        mi += 1
    print(f"Aligned {len(aligned_rows)}/{len(qa_meta)} Convs 1-5 AT1 queries.")
    return aligned_meta, aligned_rows


def build_dataset(qa_meta, dump_rows, same_sess_gain=0.4):
    X_list = []
    y_list = []
    groups = []
    q_meta_out = []
    n_feats = None

    offset = 0
    for m, r in zip(qa_meta, dump_rows):
        feats = r["feats"]
        if not feats:
            continue
        if n_feats is None:
            n_feats = len(feats[0])
        pool_ids = r["pool_ids"]
        pool_sess = r["pool_sess"]
        gs = m["gold_s"]
        gt = m["gold_t"]
        n = len(pool_ids)
        arr = np.asarray(feats, dtype=np.int32)
        ys = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if pool_ids[i] in gt:
                ys[i] = 15.0
            elif pool_sess[i] in gs and same_sess_gain > 0.0 and arr[i, 24] == 0 and arr[i, 25] == 0:
                ys[i] = same_sess_gain
        X_list.append(arr)
        y_list.append(ys)
        groups.append((offset, offset + n))
        q_meta_out.append(
            {
                "conv": m["conv"],
                "cat": m["cat"],
                "pool_ids": pool_ids,
                "pool_sess": pool_sess,
                "gold_s": gs,
                "gold_t": gt,
            }
        )
        offset += n

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y, groups, q_meta_out, n_feats


def make_discounts(max_n):
    d = np.zeros(max_n, dtype=np.float64)
    for r in range(max_n):
        if r == 0:
            d[r] = 3.0
        elif r < 5:
            d[r] = 1.0 / math.log2(r + 2.0)
        else:
            d[r] = 0.20 / math.log2(r + 2.0)
    return d


def compute_lambdas(scores, y, groups, active_q_indices, discounts, sigma=1.0):
    N = scores.shape[0]
    g = np.zeros(N, dtype=np.float64)
    h = np.zeros(N, dtype=np.float64)

    for qi in active_q_indices:
        s0, s1 = groups[qi]
        yq = y[s0:s1]
        if np.max(yq) <= 0.0:
            continue
        sq = scores[s0:s1]
        n = s1 - s0
        disc = discounts[:n]

        ideal_y = np.sort(yq)[::-1]
        idcg = float(np.dot(ideal_y, disc))
        if idcg <= 1e-12:
            continue
        inv_idcg = 1.0 / idcg

        order = np.argsort(-sq, kind="mergesort")
        ranks = np.empty(n, dtype=np.int64)
        ranks[order] = np.arange(n)

        # Pairwise matrix only over pairs where y_i > y_j
        # (n <= 56 so n x n vectorized numpy ops take microseconds!)
        dy = yq[:, None] - yq[None, :]
        mask = dy > 1e-9
        if not np.any(mask):
            continue
        dd = np.abs(disc[ranks[:, None]] - disc[ranks[None, :]])
        delta_ndcg = np.abs(dy) * dd * inv_idcg

        ds = sigma * (sq[:, None] - sq[None, :])
        ds = np.clip(ds, -30.0, 30.0)
        rho = 1.0 / (1.0 + np.exp(ds))  # P(j beats i)

        lam_mat = np.where(mask, -sigma * delta_ndcg * rho, 0.0)
        hess_mat = np.where(mask, (sigma * sigma) * delta_ndcg * rho * (1.0 - rho), 0.0)

        # g_i += sum_j lam_ij; g_j -= sum_i lam_ij
        gq = lam_mat.sum(axis=1) - lam_mat.sum(axis=0)
        hq = hess_mat.sum(axis=1) + hess_mat.sum(axis=0)
        g[s0:s1] = gq
        h[s0:s1] = hq

    return g, h


def build_bins(X, max_bins=63):
    n_samples, n_feats = X.shape
    thresholds = []
    X_bin = np.zeros((n_samples, n_feats), dtype=np.uint8)
    for f in range(n_feats):
        col = X[:, f]
        uniq = np.unique(col)
        if len(uniq) <= 1:
            thresholds.append(np.array([], dtype=np.int32))
            continue
        if len(uniq) - 1 <= max_bins:
            th = uniq[:-1].astype(np.int32)
        else:
            qs = np.linspace(0.0, 100.0, max_bins + 2)[1:-1]
            cand = np.unique(np.percentile(col, qs, method="lower").astype(np.int32))
            cand = cand[cand < uniq[-1]]
            th = cand[:max_bins]
        thresholds.append(th)
        X_bin[:, f] = np.searchsorted(th, col, side="left").astype(np.uint8)
    return X_bin, thresholds


def fit_tree(
    X_bin,
    thresholds,
    g,
    h,
    row_indices,
    max_depth=4,
    min_child_samples=20,
    min_child_hess=1e-3,
    l2_reg=8.0,
    colsample=0.85,
    rng=None,
):
    """Fits one depth-`max_depth` tree and returns list of (feat, left, right, thresh_or_leaf_f64)."""
    n_feats = X_bin.shape[1]
    nodes = []  # temporary dicts before flattening

    def build_node(idxs, depth):
        node_id = len(nodes)
        G = float(g[idxs].sum())
        H = float(h[idxs].sum())
        leaf_val = -G / (H + l2_reg) if (H + l2_reg) > 1e-12 else 0.0
        nodes.append({"leaf": True, "val": leaf_val})

        if depth >= max_depth or len(idxs) < 2 * min_child_samples:
            return node_id

        parent_gain = (G * G) / (H + l2_reg)
        best_gain = 1e-9
        best_split = None  # (f, bin_idx, thresh_int)

        if colsample < 1.0 and rng is not None:
            feat_order = rng.choice(n_feats, size=max(1, int(n_feats * colsample)), replace=False)
        else:
            feat_order = range(n_feats)

        g_sub = g[idxs]
        h_sub = h[idxs]

        for f in feat_order:
            th = thresholds[f]
            nb = len(th)
            if nb == 0:
                continue
            bins = X_bin[idxs, f]
            n_buckets = nb + 1
            cnt_b = np.bincount(bins, minlength=n_buckets)
            g_b = np.bincount(bins, weights=g_sub, minlength=n_buckets)
            h_b = np.bincount(bins, weights=h_sub, minlength=n_buckets)

            cnt_L = np.cumsum(cnt_b[:-1])
            cnt_R = len(idxs) - cnt_L
            g_L = np.cumsum(g_b[:-1])
            h_L = np.cumsum(h_b[:-1])
            g_R = G - g_L
            h_R = H - h_L

            valid = (
                (cnt_L >= min_child_samples)
                & (cnt_R >= min_child_samples)
                & (h_L >= min_child_hess)
                & (h_R >= min_child_hess)
            )
            if not np.any(valid):
                continue

            gains = (g_L * g_L) / (h_L + l2_reg) + (g_R * g_R) / (h_R + l2_reg) - parent_gain
            gains = np.where(valid, gains, -1.0)
            bi = int(np.argmax(gains))
            if gains[bi] > best_gain:
                best_gain = float(gains[bi])
                best_split = (int(f), bi, int(th[bi]))

        if best_split is None:
            return node_id

        f_best, bi_best, th_best = best_split
        left_mask = X_bin[idxs, f_best] <= bi_best
        left_idxs = idxs[left_mask]
        right_idxs = idxs[~left_mask]

        left_id = build_node(left_idxs, depth + 1)
        right_id = build_node(right_idxs, depth + 1)
        nodes[node_id] = {
            "leaf": False,
            "feat": f_best,
            "left": left_id,
            "right": right_id,
            "thresh": th_best,
        }
        return node_id

    build_node(row_indices, 0)
    return nodes


def eval_tree_vec(nodes, X, scale=1.0):
    N = X.shape[0]
    out = np.zeros(N, dtype=np.float64)

    def walk(nid, mask_idx):
        if len(mask_idx) == 0:
            return
        nd = nodes[nid]
        if nd["leaf"]:
            out[mask_idx] = nd["val"] * scale
            return
        go_left = X[mask_idx, nd["feat"]] <= nd["thresh"]
        walk(nd["left"], mask_idx[go_left])
        walk(nd["right"], mask_idx[~go_left])

    walk(0, np.arange(N, dtype=np.int64))
    return out


def eval_int_trees(int_trees, X):
    N = X.shape[0]
    scores = np.zeros(N, dtype=np.int64)
    for nodes in int_trees:
        def walk(nid, idxs):
            if len(idxs) == 0:
                return
            feat, left, right, tol = nodes[nid]
            if feat == 255:
                scores[idxs] += tol
                return
            go_left = X[idxs, feat] <= tol
            walk(left, idxs[go_left])
            walk(right, idxs[~go_left])

        walk(0, np.arange(N, dtype=np.int64))
    return scores


def quantize_trees(raw_trees, lr, q16_scale=65536.0):
    int_trees = []
    for nodes in raw_trees:
        qnodes = []
        for nd in nodes:
            if nd["leaf"]:
                w = int(round(nd["val"] * lr * q16_scale))
                w = max(-2_000_000, min(2_000_000, w))
                qnodes.append((255, 0, 0, w))
            else:
                qnodes.append((nd["feat"], nd["left"], nd["right"], int(nd["thresh"])))
        int_trees.append(qnodes)
    return int_trees


def eval_metrics(scores, groups, q_meta, q_indices):
    t1 = t5 = s1 = 0
    cat_t1 = defaultdict(int)
    cat_s1 = defaultdict(int)
    cat_n = defaultdict(int)
    for qi in q_indices:
        s0, s1_end = groups[qi]
        sq = scores[s0:s1_end]
        m = q_meta[qi]
        cat = m["cat"]
        cat_n[cat] += 1
        order = np.argsort(-sq, kind="mergesort")
        top_ids = [m["pool_ids"][i] for i in order[:5]]
        top_s = m["pool_sess"][order[0]]
        if top_ids and top_ids[0] in m["gold_t"]:
            t1 += 1
            cat_t1[cat] += 1
        if any(tid in m["gold_t"] for tid in top_ids):
            t5 += 1
        if top_s in m["gold_s"]:
            s1 += 1
            cat_s1[cat] += 1
    n = max(1, len(q_indices))
    return {
        "n": len(q_indices),
        "t1": 100.0 * t1 / n,
        "t5": 100.0 * t5 / n,
        "s1": 100.0 * s1 / n,
        "cat_t1": {c: 100.0 * cat_t1[c] / max(1, cat_n[c]) for c in sorted(cat_n)},
        "cat_s1": {c: 100.0 * cat_s1[c] / max(1, cat_n[c]) for c in sorted(cat_n)},
    }


def train_lambdamart(
    X,
    X_bin,
    thresholds,
    y,
    groups,
    train_q_indices,
    n_trees=64,
    max_depth=4,
    lr=0.07,
    l2_reg=10.0,
    min_child_samples=20,
    colsample=0.80,
    seed=42,
):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    scores = np.zeros(N, dtype=np.float64)
    discounts = make_discounts(128)

    train_rows = np.concatenate(
        [np.arange(groups[qi][0], groups[qi][1], dtype=np.int64) for qi in train_q_indices]
    )

    raw_trees = []
    for _t in range(n_trees):
        g, h = compute_lambdas(scores, y, groups, train_q_indices, discounts, sigma=1.0)
        # Filter to rows with non-zero hessian for cleaner splits
        nz = train_rows[h[train_rows] > 1e-12]
        if len(nz) < 2 * min_child_samples:
            nz = train_rows
        tree = fit_tree(
            X_bin,
            thresholds,
            g,
            h,
            nz,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            l2_reg=l2_reg,
            colsample=colsample,
            rng=rng,
        )
        raw_trees.append(tree)
        scores += eval_tree_vec(tree, X, scale=lr)

    # Auto-calibrate Q16 leaf scale so the ensemble score std-dev sits at ~55_000 Q16 units,
    # matching the exact Q16 scale expected by `AT1_SESS_SYNC_MARGIN_Q16 = 32768` and post-GBDT margins!
    tr_std = float(np.std(scores[train_rows]))
    q16_scale = 55000.0 / max(tr_std, 1e-4)
    int_trees = quantize_trees(raw_trees, lr, q16_scale=q16_scale)
    return int_trees


def serialize_and_hash(int_trees):
    roots = []
    flat = []
    for t in int_trees:
        roots.append(len(flat))
        flat.extend(t)

    buf = bytearray()
    buf.extend(struct.pack("<I", len(roots)))
    for r in roots:
        buf.extend(struct.pack("<H", r))
    buf.extend(struct.pack("<I", len(flat)))
    for feat, left, right, tol in flat:
        buf.extend(struct.pack("<BBBBi", feat, left, right, 0, tol))
    sha = hashlib.sha256(bytes(buf)).hexdigest()
    return roots, flat, sha


def write_weights_to_rs(roots, flat, sha, n_feats):
    src = open(AT1_RS, "r", encoding="utf-8").read()
    # Update N_AT1_FEATS if needed
    src = re.sub(
        r"pub const N_AT1_FEATS: usize = \d+;",
        f"pub const N_AT1_FEATS: usize = {n_feats};",
        src,
        count=1,
    )
    roots_str = ", ".join(str(r) for r in roots)
    nodes_lines = []
    for feat, left, right, tol in flat:
        feat_s = "u8::MAX" if feat == 255 else str(feat)
        nodes_lines.append(
            f"    GbdtNode {{ feat: {feat_s}, left: {left}, right: {right}, thresh_or_leaf: {tol} }},"
        )
    nodes_block = "\n".join(nodes_lines)

    new_block = (
        "// === BEGIN FROZEN GBDT WEIGHTS (fitted strictly on Convs 1-5 disjoint split, verified by SHA-256 test) ===\n"
        "/// SHA-256 hex digest of the canonical byte encoding of `AT1_TREE_ROOTS` + `AT1_TREE_NODES`.\n"
        "#[cfg(test)]\n"
        f'pub const AT1_GBDT_SHA256: &str =\n    "{sha}";\n\n'
        "/// Start index of each tree inside `AT1_TREE_NODES`.\n"
        f"pub const AT1_TREE_ROOTS: &[u16] = &[{roots_str}];\n\n"
        "/// Frozen GBDT node table.\n"
        "pub const AT1_TREE_NODES: &[GbdtNode] = &[\n"
        f"{nodes_block}\n"
        "];\n\n"
        "/// Minimum integer GBDT margin required for top-turn session concordance to promote `sessions[0]`.\n"
        "pub const AT1_SESS_SYNC_MARGIN_Q16: i64 = 32768;\n"
        "// === END FROZEN GBDT WEIGHTS ==="
    )

    pattern = re.compile(
        r"// === BEGIN FROZEN GBDT WEIGHTS.*?// === END FROZEN GBDT WEIGHTS ===",
        re.DOTALL,
    )
    assert pattern.search(src), "Could not find FROZEN GBDT WEIGHTS markers in at1_scorer_q16.rs"
    updated = pattern.sub(new_block, src, count=1)
    open(AT1_RS, "w", encoding="utf-8", newline="\n").write(updated)
    print(f"Wrote {len(roots)} trees ({len(flat)} nodes), N_AT1_FEATS={n_feats}, sha256={sha}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=DEFAULT_BIN)
    ap.add_argument("--dump", default=DEFAULT_DUMP)
    ap.add_argument("--reuse-dump", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--n-trees", type=int, default=64)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    t0 = time.time()
    if args.reuse_dump and os.path.exists(args.dump):
        data = json.load(open(DATA_PATH, "r", encoding="utf-8"))[:5]
        qa_meta = []
        for ci, conv in enumerate(data):
            conv_obj = conv["conversation"]
            n_sess = 0
            while f"session_{n_sess + 1}" in conv_obj:
                n_sess += 1
            for qa in conv["qa"]:
                gs, gt = parse_evidence(qa.get("evidence", []))
                if gs or gt:
                    qa_meta.append(
                        {
                            "conv": ci,
                            "n_sessions": n_sess,
                            "cat": int(qa["category"]),
                            "q": qa["question"],
                            "gold_s": gs,
                            "gold_t": gt,
                        }
                    )
        raw_rows = [json.loads(l) for l in open(args.dump, "r", encoding="utf-8") if l.strip()]
        qa_meta, dump_rows = align_dump_to_meta(qa_meta, raw_rows)
    else:
        print(f"Dumping Convs 1-5 features using {args.bin} -> {args.dump} ...")
        qa_meta, dump_rows = dump_convs_1_5_features(args.bin, args.dump)
    print(f"Loaded {len(dump_rows)} Convs 1-5 queries in {time.time() - t0:.1f}s")

    # Hyperparameter sweep via 5-fold Leave-One-Conversation-Out CV strictly inside Convs 1-5
    configs = [
        {"lr": 0.08, "l2": 4.0, "mcs": 16, "ss_gain": 0.0, "colsample": 0.75},
        {"lr": 0.08, "l2": 6.0, "mcs": 18, "ss_gain": 0.25, "colsample": 0.75},
        {"lr": 0.07, "l2": 3.0, "mcs": 14, "ss_gain": 0.15, "colsample": 0.70},
    ]

    best_cfg = None
    best_cv_score = -1.0

    for cfg in configs:
        X, y, groups, q_meta_out, n_feats = build_dataset(qa_meta, dump_rows, same_sess_gain=cfg["ss_gain"])
        X_bin, thresholds = build_bins(X, max_bins=63)
        oof_scores = np.zeros(X.shape[0], dtype=np.int64)
        for val_conv in range(5):
            tr_q = [i for i, m in enumerate(q_meta_out) if m["conv"] != val_conv]
            va_q = [i for i, m in enumerate(q_meta_out) if m["conv"] == val_conv]
            trees = train_lambdamart(
                X, X_bin, thresholds, y, groups, tr_q,
                n_trees=args.n_trees, max_depth=args.max_depth,
                lr=cfg["lr"], l2_reg=cfg["l2"], min_child_samples=cfg["mcs"],
                colsample=cfg["colsample"], seed=args.seed + val_conv,
            )
            va_scores = eval_int_trees(trees, X)
            for qi in va_q:
                s0, s1 = groups[qi]
                oof_scores[s0:s1] = va_scores[s0:s1]
        m = eval_metrics(oof_scores, groups, q_meta_out, list(range(len(q_meta_out))))
        obj = m["t1"] + 0.5 * m["t5"] + 0.5 * m["s1"]
        print(
            f"5-fold LOCO-CV (Convs 1-5) cfg={cfg} -> "
            f"T@1={m['t1']:.2f}% T@5={m['t5']:.2f}% S@1={m['s1']:.2f}% obj={obj:.2f}"
        )
        print(
            "   by cat T@1:",
            {CAT_NAMES[c]: f"{v:.2f}%" for c, v in m["cat_t1"].items()},
        )
        if obj > best_cv_score:
            best_cv_score = obj
            best_cfg = cfg

    print(f"\nSelected best Convs 1-5 LOCO-CV config: {best_cfg}")
    X, y, groups, q_meta_out, n_feats = build_dataset(qa_meta, dump_rows, same_sess_gain=best_cfg["ss_gain"])
    X_bin, thresholds = build_bins(X, max_bins=63)
    all_q = list(range(len(q_meta_out)))
    final_trees = train_lambdamart(
        X, X_bin, thresholds, y, groups, all_q,
        n_trees=args.n_trees, max_depth=args.max_depth,
        lr=best_cfg["lr"], l2_reg=best_cfg["l2"], min_child_samples=best_cfg["mcs"],
        colsample=best_cfg["colsample"], seed=args.seed,
    )
    in_scores = eval_int_trees(final_trees, X)
    im = eval_metrics(in_scores, groups, q_meta_out, all_q)
    print(f"Final Convs 1-5 fit (in-sample): T@1={im['t1']:.2f}% T@5={im['t5']:.2f}% S@1={im['s1']:.2f}%")

    # Print feature split histogram so we verify new features (f[24], f[25], f[32], f[36..48]) are actively used!
    feat_counts = defaultdict(int)
    for t in final_trees:
        for feat, _, _, _ in t:
            if feat != 255:
                feat_counts[feat] += 1
    top_f = sorted(feat_counts.items(), key=lambda kv: -kv[1])
    print(f"Total splits: {sum(feat_counts.values())}; top features: {top_f[:20]}")
    print(f"Key discourse splits: f[24]={feat_counts[24]} f[25]={feat_counts[25]} f[32]={feat_counts[32]} f[36..47]={ {k: feat_counts[k] for k in range(36, n_feats)} }")

    roots, flat, sha = serialize_and_hash(final_trees)
    print(f"Serialized {len(roots)} trees, {len(flat)} nodes, SHA-256={sha}")
    if args.write:
        write_weights_to_rs(roots, flat, sha, n_feats)


if __name__ == "__main__":
    main()
