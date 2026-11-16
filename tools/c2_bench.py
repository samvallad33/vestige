#!/usr/bin/env python3
"""C2 three-bench protocol and RULE 1 stage funnel.

Scoreboard (the 0-loss gate):
  LME_S   n=470 non-abstention   session Any@1/5/10 All@5/All@10, turn Any@1/5/10 All@5
  LoCoMo  n=1536 cat 1-4 with evidence   turn Any@1/5/10, session Any@1/5
  BEAM    n=355 non-empty source_chat_ids   turn Any@1/5/10 All@5/All@10, session Any@1/5/10

Ingestion matches benchmarks/run_stage1_retrieval.mjs (LME, LoCoMo) and
benchmarks/run_recall_suite.mjs (BEAM). Labels never enter the rank request.

RULE 1 fields are written by crates/vestige-core/src/retrieval.rs and checked
here against the response (see verify_funnel_contract):

  corpus            gold id was assigned by this harness's ingestion
  pool50_ids        scene_pool50(), immediately before submodular selection
  pre_rank_ids      list cloned immediately before apply_step11_kemeny_wh_slot_voter
  post_kemeny_ids   list cloned immediately after rank1_gatekeeper_keep_rsb_head
  final_ids         RankResult.turns. @1 is final_ids[0]
  final_session_ids immutable final_session_indices. Session pre_rank@5 and
                    post_kemeny@5 are this list's top 5: nothing takes
                    `&mut final_session_indices`.

Usage:
  python3 tools/c2_bench.py --out /tmp/c2-runs/step1
  python3 tools/c2_bench.py --diff /tmp/c2-runs/step1/questions.jsonl /tmp/c2-runs/step2/questions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL = ROOT / "crates" / "vestige-core" / "src" / "retrieval.rs"
DIA_RE = re.compile(r"^D\d+:\d+$")

TURN_KEYS = {
    "LME_S": ("1", "5", "10", "all5"),
    "LoCoMo": ("1", "5", "10"),
    "BEAM": ("1", "5", "10", "all5", "all10"),
}
SESS_KEYS = {
    "LME_S": ("1", "5", "10", "all5", "all10"),
    "LoCoMo": ("1", "5"),
    "BEAM": ("1", "5", "10"),
}
FUNNEL_KEYS = ("corpus", "pool50", "pre_rank5", "post_kemeny5", "at1")


def verify_funnel_contract() -> None:
    """Fail if the dump writers or the session-identity claim are not in the source."""
    src = RETRIEVAL.read_text(encoding="utf-8")
    required = [
        "fn scene_pool50(",
        "fn session_pool50(",
        "fn maybe_dump_funnel(",
        "let funnel_pre_rank: Vec<usize> = if funnel_on {",
        "let funnel_post_kemeny: Vec<usize> = if funnel_on {",
        "apply_step11_kemeny_wh_slot_voter",
        "rank1_gatekeeper_keep_rsb_head(ix, &q.text, &mut final_turn_indices, &t_scene);",
        '"pool50_ids"',
        '"pre_rank_ids"',
        '"post_kemeny_ids"',
        '"final_ids"',
        '"final_session_ids"',
        '"corpus_turns"',
    ]
    missing = [s for s in required if s not in src]
    if missing:
        raise SystemExit("funnel contract missing from retrieval.rs: " + ", ".join(missing))
    if "&mut final_session_indices" in src:
        raise SystemExit(
            "session pre_rank@5 == post_kemeny@5 claim is false: "
            "a seat now takes &mut final_session_indices"
        )
    # The pre_rank clone must sit textually before the Kemeny voter, and the
    # post_kemeny clone must sit textually after the gatekeeper call.
    pre = src.find("let funnel_pre_rank:")
    kem = src.find("apply_step11_kemeny_wh_slot_voter(ix, &q.text")
    gate = src.find("rank1_gatekeeper_keep_rsb_head(ix, &q.text")
    post = src.find("let funnel_post_kemeny:")
    pool = src.find("scene_pool50(&t_scene, &turn_ts)")
    sub = src.find("let mut final_turn_indices: Vec<usize> = if p.enable_submodular")
    if not (0 <= pool < sub < pre < kem < gate < post):
        raise SystemExit(
            f"funnel stage order is wrong in source: pool={pool} sub={sub} "
            f"pre={pre} kemeny={kem} gate={gate} post={post}"
        )


def find_bin(explicit: str | None = None) -> str:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"retrieval_bench not found: {explicit}")
        return str(p)
    for rel in ("target/release/retrieval_bench", "target/debug/retrieval_bench"):
        p = ROOT / rel
        if p.exists():
            return str(p)
    raise SystemExit("retrieval_bench not built. cargo build --release -p vestige-core --bin retrieval_bench")


class Worker:
    def __init__(self, bin_path: str, dump_path: Path):
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text("", encoding="utf-8")
        self.dump_path = dump_path
        self.offset = 0
        env = os.environ.copy()
        env["VESTIGE_FUNNEL_DUMP"] = str(dump_path)
        env.pop("VESTIGE_REPLAY_DUMP", None)
        self.proc = subprocess.Popen(
            [bin_path, "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self.lock = threading.Lock()

    def rank(self, sessions, queries):
        payload = json.dumps({"sessions": sessions, "queries": queries, "top_k": 10}).encode()
        with self.lock:
            assert self.proc.stdin is not None and self.proc.stdout is not None
            self.proc.stdin.write(payload + b"\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                err = b""
                if self.proc.stderr:
                    err = self.proc.stderr.read()
                raise RuntimeError(f"retrieval_bench closed: {err[-2000:]!r}")
            resp = json.loads(line)
            if isinstance(resp, dict) and resp.get("error"):
                raise RuntimeError(resp["error"])
            dumps = self._read_dumps(len(queries))
        return resp, dumps

    def _read_dumps(self, n: int):
        with self.dump_path.open("r", encoding="utf-8") as f:
            f.seek(self.offset)
            extra = f.read()
            self.offset = f.tell()
        rows = [json.loads(l) for l in extra.splitlines() if l.strip()]
        if len(rows) != n:
            raise RuntimeError(f"funnel dump lines {len(rows)} != queries {n}")
        return rows

    def close(self):
        if self.proc.stdin:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def any_at(gold, ids, k):
    if not gold:
        return 0
    window = ids[:k]
    return 1 if any(g in window for g in gold) else 0


def all_at(gold, ids, k):
    if not gold:
        return 0
    window = set(ids[:k])
    return 1 if all(g in window for g in gold) else 0


def first_rank(gold, ids):
    gset = set(gold)
    for i, x in enumerate(ids):
        if x in gset:
            return i
    return -1


def hits(bench, gold_turns, turn_ids, gold_sess, sess_ids):
    th = {}
    sh = {}
    for k in TURN_KEYS[bench]:
        if k.startswith("all"):
            th[k] = all_at(gold_turns, turn_ids, int(k[3:]))
        else:
            th[k] = any_at(gold_turns, turn_ids, int(k))
    for k in SESS_KEYS[bench]:
        if k.startswith("all"):
            sh[k] = all_at(gold_sess, sess_ids, int(k[3:]))
        else:
            sh[k] = any_at(gold_sess, sess_ids, int(k))
    return th, sh


def funnel_any(gold, corpus_ids, pool_ids, pre_ids, post_ids, final_ids):
    return {
        "corpus": any_at(gold, list(corpus_ids), 10**9),
        "pool50": any_at(gold, pool_ids, 50),
        "pre_rank5": any_at(gold, pre_ids, 5),
        "post_kemeny5": any_at(gold, post_ids, 5),
        "at1": any_at(gold, final_ids, 1),
    }


def check_row(dump, resp_turns, resp_sess, n_turns, n_sess):
    need = (
        "pool50_ids",
        "session_pool50_ids",
        "pre_rank_ids",
        "post_kemeny_ids",
        "final_ids",
        "final_session_ids",
        "corpus_turns",
        "corpus_sessions",
    )
    missing = [k for k in need if k not in dump]
    if missing:
        raise RuntimeError(f"funnel record missing {missing}")
    if dump["final_ids"] != resp_turns:
        raise RuntimeError(
            "final_ids diverges from RankResponse.turns: "
            f"dump={dump['final_ids'][:3]} resp={resp_turns[:3]}"
        )
    if dump["final_session_ids"] != resp_sess:
        raise RuntimeError(
            "final_session_ids diverges from RankResponse.sessions: "
            f"dump={dump['final_session_ids'][:3]} resp={resp_sess[:3]}"
        )
    if dump["corpus_turns"] != n_turns or dump["corpus_sessions"] != n_sess:
        raise RuntimeError(
            f"corpus size dump={dump['corpus_turns']}/{dump['corpus_sessions']} "
            f"sent={n_turns}/{n_sess}"
        )


def record(bench, qid, cat, gate, gold_turns, gold_sess, corpus_turns, corpus_sess, resp_row, dump, n_turns, n_sess):
    turns = [t["id"] for t in resp_row.get("turns") or []]
    sess = [s["id"] for s in resp_row.get("sessions") or []]
    check_row(dump, turns, sess, n_turns, n_sess)
    # Session pre_rank@5 and post_kemeny@5 are final_session_ids[:5].
    # Code-verified: final_session_indices is never mutably borrowed.
    sess_pre = dump["final_session_ids"]
    th, sh = hits(bench, gold_turns, turns, gold_sess, sess)
    return {
        "bench": bench,
        "id": qid,
        "cat": cat,
        "gate": 1 if gate else 0,
        "turn_hit": th,
        "sess_hit": sh,
        "turn_rank": first_rank(gold_turns, turns),
        "sess_rank": first_rank(gold_sess, sess),
        "funnel_turn": funnel_any(
            gold_turns, corpus_turns, dump["pool50_ids"], dump["pre_rank_ids"], dump["post_kemeny_ids"], dump["final_ids"]
        ),
        "funnel_sess": funnel_any(
            gold_sess,
            corpus_sess,
            dump["session_pool50_ids"],
            sess_pre,
            sess_pre,
            dump["final_session_ids"],
        ),
        "head_turn": turns[0] if turns else "",
        "head_sess": sess[0] if sess else "",
        "top_turns": turns[:10],
        "top_sess": sess[:10],
        "gold_turns": list(gold_turns),
        "gold_sess": list(gold_sess),
        "pool_has_turn": any_at(gold_turns, dump["pool50_ids"], 50),
        "pre_has_turn": any_at(gold_turns, dump["pre_rank_ids"], 5),
        "post_has_turn": any_at(gold_turns, dump["post_kemeny_ids"], 5),
        "seat_fires": int(dump.get("queried_speaker_fires") or 0),
    }


def tally(rows, bench):
    gate = [r for r in rows if r["bench"] == bench and r["gate"]]
    acc = {
        "n": len(gate),
        "turn": {k: 0 for k in TURN_KEYS[bench]},
        "sess": {k: 0 for k in SESS_KEYS[bench]},
        "funnel_turn": {k: 0 for k in FUNNEL_KEYS},
        "funnel_sess": {k: 0 for k in FUNNEL_KEYS},
    }
    cats = defaultdict(lambda: {"n": 0, "turn": defaultdict(int), "sess": defaultdict(int)})
    for r in gate:
        for k, v in r["turn_hit"].items():
            acc["turn"][k] += v
        for k, v in r["sess_hit"].items():
            acc["sess"][k] += v
        for k, v in r["funnel_turn"].items():
            acc["funnel_turn"][k] += v
        for k, v in r["funnel_sess"].items():
            acc["funnel_sess"][k] += v
        c = cats[str(r["cat"])]
        c["n"] += 1
        for k, v in r["turn_hit"].items():
            c["turn"][k] += v
        for k, v in r["sess_hit"].items():
            c["sess"][k] += v
    return acc, cats


def pct(n, d):
    return (100.0 * n / d) if d else 0.0


def fmt_counts(keys, bag, n):
    parts = []
    for k in keys:
        parts.append(f"{k}={bag[k]}/{n} ({pct(bag[k], n):.2f}%)")
    return " ".join(parts)


def print_halves(rows):
    """Even/odd gate index in harness order.

    Frozen before any outcome was read: index 0, 2, 4... is the design half
    and index 1, 3, 5... is the confirmation half. This is not a ranker input.
    """
    print("===== EVEN GATE INDEX (design) / ODD GATE INDEX (confirmation) =====")
    print("split: 0-based index among gate rows in harness order, frozen before the run")
    for bench in ("LME_S", "LoCoMo", "BEAM"):
        gate = [r for r in rows if r["bench"] == bench and r["gate"]]
        for name, half in (("even", gate[0::2]), ("odd", gate[1::2])):
            n = len(half)
            t1 = sum(r["turn_hit"].get("1", 0) for r in half)
            fires = sum(int(r.get("seat_fires") or 0) for r in half)
            fired = sum(1 for r in half if int(r.get("seat_fires") or 0) > 0)
            print(f"  {bench} {name} n={n} turn@1={t1}/{n} swaps={fires} queries={fired}")


def print_scoreboard(rows):
    print("===== RULE 1 / RULE 2 STAGE FUNNEL + C2 SCOREBOARD =====")
    print("session pre_rank@5 == session post_kemeny@5 == final_session_ids[:5]")
    print("reason: final_session_indices is let-bound and never mutably borrowed")
    for bench in ("LME_S", "LoCoMo", "BEAM"):
        acc, cats = tally(rows, bench)
        n = acc["n"]
        print(f"\n--- {bench} gate n={n} ---")
        print("  TURN  " + fmt_counts(TURN_KEYS[bench], acc["turn"], n))
        print("  SESS  " + fmt_counts(SESS_KEYS[bench], acc["sess"], n))
        print("  FUNNEL turn any-gold " + fmt_counts(FUNNEL_KEYS, acc["funnel_turn"], n))
        print("  FUNNEL sess any-gold " + fmt_counts(FUNNEL_KEYS, acc["funnel_sess"], n))
        fires = sum(int(r.get("seat_fires") or 0) for r in gate)
        fired = sum(1 for r in gate if int(r.get("seat_fires") or 0) > 0)
        print(f"  queried_speaker_reply swaps={fires} queries={fired}/{n}")
        for cat, c in sorted(cats.items(), key=lambda kv: str(kv[0])):
            print(
                f"    [{cat}] n={c['n']}  TURN {fmt_counts(TURN_KEYS[bench], c['turn'], c['n'])}"
            )
            print(
                f"    [{cat}] n={c['n']}  SESS {fmt_counts(SESS_KEYS[bench], c['sess'], c['n'])}"
            )


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ingest_lme(item):
    sessions = []
    gold_turns = []
    corpus_turns = []
    for i, sess in enumerate(item["haystack_sessions"]):
        sid = str(item["haystack_session_ids"][i])
        turns = []
        for j, t in enumerate(sess):
            tid = f"{sid}#{j}"
            turns.append({
                "id": tid,
                "role": str(t.get("role") or ""),
                "content": str(t.get("content") or ""),
            })
            corpus_turns.append(tid)
            if t.get("has_answer") is True:
                gold_turns.append(tid)
        sessions.append({
            "id": sid,
            "date": str(item["haystack_dates"][i] or ""),
            "turns": turns,
        })
    gold_sess = [str(s) for s in (item.get("answer_session_ids") or [])]
    q = {"text": str(item["question"]), "date": str(item.get("question_date") or "")}
    return sessions, q, gold_turns, gold_sess, corpus_turns


def run_lme(worker, limit):
    data = load_json(ROOT / "benchmarks" / "data" / "longmemeval_s_cleaned.json")
    if limit:
        data = data[:limit]
    rows = []
    for i, item in enumerate(data):
        sessions, q, gold_turns, gold_sess, corpus_turns = ingest_lme(item)
        n_turns = sum(len(s["turns"]) for s in sessions)
        resp, dumps = worker.rank(sessions, [q])
        qid = str(item.get("question_id"))
        gate = not qid.endswith("_abs")
        rows.append(record(
            "LME_S", qid, item.get("question_type"), gate,
            gold_turns, gold_sess, corpus_turns, [s["id"] for s in sessions],
            resp["results"][0], dumps[0], n_turns, len(sessions),
        ))
        if (i + 1) % 25 == 0 or i + 1 == len(data):
            print(f"[LME {i + 1}/{len(data)}]", flush=True)
    return rows


def evidence_ids(raw):
    parts = []
    if raw is None:
        return parts
    seq = raw if isinstance(raw, list) else [raw]
    for e in seq:
        for tok in re.split(r"[;,\s]+", str(e)):
            if DIA_RE.match(tok):
                parts.append(tok)
    return parts


def ingest_locomo(conv):
    c = conv["conversation"]
    sessions = []
    corpus_turns = []
    s = 1
    while c.get(f"session_{s}"):
        turns = []
        for t in c[f"session_{s}"]:
            if not t.get("text") or not t.get("dia_id"):
                continue
            blip = f"\n[BLIP caption] {t['blip_caption']}" if t.get("blip_caption") else ""
            tid = str(t["dia_id"])
            turns.append({"id": tid, "role": str(t.get("speaker") or ""), "content": f"{t['text']}{blip}"})
            corpus_turns.append(tid)
        sessions.append({
            "id": f"D{s}",
            "date": str(c.get(f"session_{s}_date_time") or ""),
            "turns": turns,
        })
        s += 1
    queries = [{"text": str(qa.get("question") or ""), "date": ""} for qa in conv["qa"]]
    return sessions, queries, corpus_turns


def run_locomo(worker, limit):
    data = load_json(ROOT / "benchmarks" / "data" / "locomo10.json")
    rows = []
    seen = 0
    for conv in data:
        sessions, queries, corpus_turns = ingest_locomo(conv)
        if limit and seen >= limit:
            break
        if limit:
            queries = queries[: max(0, limit - seen)]
            qa_slice = conv["qa"][: len(queries)]
        else:
            qa_slice = conv["qa"]
        if not queries:
            continue
        n_turns = sum(len(s["turns"]) for s in sessions)
        resp, dumps = worker.rank(sessions, queries)
        corpus_sess = [s["id"] for s in sessions]
        for qi, qa in enumerate(qa_slice):
            gold_turns = evidence_ids(qa.get("evidence"))
            gold_sess = sorted(set(g.split(":")[0] for g in gold_turns))
            cat = qa.get("category")
            gate = bool(gold_turns) and cat != 5
            rows.append(record(
                "LoCoMo", f"{conv.get('sample_id', 'conv')}:{qi}", cat, gate,
                gold_turns, gold_sess, corpus_turns, corpus_sess,
                resp["results"][qi], dumps[qi], n_turns, len(sessions),
            ))
        seen += len(qa_slice)
        print(f"[LoCoMo questions {seen}]", flush=True)
    return rows


def extract_gold_ids(raw):
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        out = []
        for x in raw:
            out.extend(extract_gold_ids(x))
        return out
    if isinstance(raw, dict):
        out = []
        for x in raw.values():
            out.extend(extract_gold_ids(x))
        return out
    return [str(raw)]


def run_beam(worker, limit):
    base = ROOT / "benchmarks" / "BEAM" / "chats" / "100K"
    convs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: int(p.name))
    if limit:
        convs = convs[:limit]
    rows = []
    for ci, cdir in enumerate(convs):
        chat = load_json(cdir / "chat.json")
        qdata = load_json(cdir / "probing_questions" / "probing_questions.json")
        sessions = []
        turn_to_sess = {}
        corpus_turns = []
        for bi, batch in enumerate(chat):
            batch_anchor = batch.get("time_anchor") or ""
            subs = batch.get("turns") or []
            if subs and isinstance(subs[0], list) and subs[0] and isinstance(subs[0][0], dict):
                batch_anchor = batch_anchor or subs[0][0].get("time_anchor") or ""
            for sub_idx, sub in enumerate(subs):
                if not isinstance(sub, list) or not sub:
                    continue
                sid = f"conv{cdir.name}_b{bi}_s{sub_idx}"
                turns = []
                for t in sub:
                    if not isinstance(t, dict) or t.get("id") is None or t.get("id") == "":
                        continue
                    tid = str(t["id"])
                    turns.append({
                        "id": tid,
                        "role": t.get("role") or "user",
                        "content": t.get("content") or "",
                    })
                    turn_to_sess[tid] = sid
                    corpus_turns.append(tid)
                sessions.append({
                    "id": sid,
                    "date": (sub[0].get("time_anchor") if isinstance(sub[0], dict) else None) or batch_anchor or "",
                    "timestamp": 0,
                    "turns": turns,
                })
        entries = []
        for cat, qlist in qdata.items():
            if not isinstance(qlist, list):
                continue
            for qi, qitem in enumerate(qlist):
                gold_turns = list(dict.fromkeys(extract_gold_ids(qitem.get("source_chat_ids"))))
                if not gold_turns:
                    continue
                gold_sess = list(dict.fromkeys(turn_to_sess[g] for g in gold_turns if g in turn_to_sess))
                entries.append((f"{cdir.name}:{cat}:{qi}", cat, qitem.get("question") or "", gold_turns, gold_sess))
        if not entries:
            continue
        n_turns = sum(len(s["turns"]) for s in sessions)
        resp, dumps = worker.rank(sessions, [{"text": e[2], "date": ""} for e in entries])
        corpus_sess = [s["id"] for s in sessions]
        for ei, e in enumerate(entries):
            rows.append(record(
                "BEAM", e[0], e[1], True, e[3], e[4], corpus_turns, corpus_sess,
                resp["results"][ei], dumps[ei], n_turns, len(sessions),
            ))
        print(f"[BEAM {ci + 1}/{len(convs)} scored_so_far={sum(1 for r in rows if r['gate'])}]", flush=True)
    return rows


def diff_jsonl(before, after):
    def load(path):
        rows = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[(r["bench"], r["id"])] = r
        return rows

    a = load(before)
    b = load(after)
    losses = []
    gains = []
    keys = set(a) & set(b)
    for k in sorted(keys):
        ra, rb = a[k], b[k]
        if not ra.get("gate"):
            continue
        for side in ("turn_hit", "sess_hit"):
            for metric, va in ra[side].items():
                vb = rb[side].get(metric, 0)
                if va == 1 and vb == 0:
                    losses.append((k[0], k[1], side, metric, ra.get("cat")))
                elif va == 0 and vb == 1:
                    gains.append((k[0], k[1], side, metric, ra.get("cat")))
    print(f"compared_gate_questions={sum(1 for k in keys if a[k].get('gate'))}")
    print(f"losses={len(losses)} gains={len(gains)}")
    for row in losses:
        print("LOSS", "\t".join(map(str, row)))
    for row in gains:
        print("GAIN", "\t".join(map(str, row)))
    return 1 if losses else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/c2-runs/step1"))
    ap.add_argument("--only", choices=("lme", "locomo", "beam", "all"), default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bin", default=None, help="retrieval_bench path (default: this repo's release/debug binary)")
    ap.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"))
    args = ap.parse_args()
    if args.diff:
        sys.exit(diff_jsonl(args.diff[0], args.diff[1]))
    verify_funnel_contract()
    bin_path = find_bin(args.bin)
    args.out.mkdir(parents=True, exist_ok=True)
    dump = args.out / "funnel.jsonl"
    worker = Worker(bin_path, dump)
    rows = []
    try:
        if args.only in ("lme", "all"):
            rows.extend(run_lme(worker, args.limit))
        if args.only in ("locomo", "all"):
            rows.extend(run_locomo(worker, args.limit))
        if args.only in ("beam", "all"):
            rows.extend(run_beam(worker, args.limit))
    finally:
        worker.close()
    qpath = args.out / "questions.jsonl"
    with qpath.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    print_scoreboard(rows)
    print_halves(rows)
    summary = {}
    for bench in ("LME_S", "LoCoMo", "BEAM"):
        acc, cats = tally(rows, bench)
        summary[bench] = {
            "n": acc["n"],
            "turn": acc["turn"],
            "sess": acc["sess"],
            "funnel_turn": acc["funnel_turn"],
            "funnel_sess": acc["funnel_sess"],
            "cats": {k: {"n": v["n"], "turn": dict(v["turn"]), "sess": dict(v["sess"])} for k, v in cats.items()},
        }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {qpath}")
    print(f"binary {bin_path}")


if __name__ == "__main__":
    main()
