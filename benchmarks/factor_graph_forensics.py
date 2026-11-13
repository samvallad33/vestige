#!/usr/bin/env python3
"""Ballot forensics for the shipped factor-graph rule.

Re-ranks the questions whose turn Any@5 flipped between the PR 715 card and
the v6 card. The traced worker is the same rule with VESTIGE_FACTOR_TRACE set.
Labels are read only after both workers return.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import factor_graph_card as card  # noqa: E402


def flip(pair: dict) -> dict:
    out = dict(pair)
    out["a"], out["b"] = pair["b"], pair["a"]
    for key in ("scene", "degree", "stability", "wh", "factor", "margin"):
        out[key] = -pair[key]
    return out


def lookup(trace: dict, left: str, right: str):
    for pair in trace["pairs"]:
        if pair["a"] == left and pair["b"] == right:
            return pair
        if pair["a"] == right and pair["b"] == left:
            return flip(pair)
    return None


def edge_kinds(edges: list[str]) -> str:
    kinds = []
    for edge in edges:
        kind = edge.split(":", 1)[0]
        if kind not in kinds:
            kinds.append(kind)
    return "+".join(kinds) if kinds else "none"


def rank_of(ids: list[str], item: str) -> str:
    if item in ids:
        return str(ids.index(item) + 1)
    return "out"


def loss_ids(base_path: str, new_path: str):
    base = {row["id"]: row for row in json.load(open(base_path, encoding="utf-8"))["rows"]}
    new = {row["id"]: row for row in json.load(open(new_path, encoding="utf-8"))["rows"]}
    hurt, help_ = [], []
    for qid, row in base.items():
        if row.get("abs"):
            continue
        other = new[qid]
        if row["t5"] == 1 and other["t5"] == 0:
            hurt.append(qid)
        elif row["t5"] == 0 and other["t5"] == 1:
            help_.append(qid)
    return set(hurt), set(help_), base, new


def explain(question: dict, base_ids: list[str], result: dict, gold: set[str]) -> dict:
    trace = result["factor_trace"]
    new_ids = [turn["id"] for turn in result["turns"]]
    if trace["ranked_ids"] != new_ids:
        raise SystemExit(f"{question['id']}: trace order disagrees with the returned list")
    if trace["baseline_ids"] != base_ids:
        # The no-factor tail inside this binary should match PR 715. A mismatch
        # means the selector moved, and the card comparison is not this ballot.
        question["baseline_mismatch"] = {
            "pr715": base_ids,
            "internal": trace["baseline_ids"],
        }
    base_top = base_ids[:5]
    new_top = new_ids[:5]
    entered = [item for item in new_top if item not in base_top]
    gold_left = [item for item in base_top if item in gold and item not in new_top]
    below = {item["id"]: item for item in trace["below_cut"]}
    pairs = []
    for entrant in entered:
        admitted = below.get(entrant)
        for gone in gold_left:
            ballot = lookup(trace, entrant, gone)
            pairs.append(
                {
                    "entrant": entrant,
                    "below_cut": admitted is not None,
                    "edges": admitted["edges"] if admitted else [],
                    "kinds": edge_kinds(admitted["edges"]) if admitted else "in-window",
                    "gold": gone,
                    "gold_now": rank_of(new_ids, gone),
                    "ballot": ballot,
                }
            )
    return {
        **question,
        "entered": entered,
        "gold_left": gold_left,
        "gold_now": {item: rank_of(new_ids, item) for item in gold_left},
        "pairs": pairs,
        "base_top": base_top,
        "new_top": new_top,
    }


def tally_line(ballot: dict | None) -> str:
    if ballot is None:
        return "no pair"
    counted = "yes" if ballot["degree_counted"] else "silent"
    return (
        f"scene {ballot['scene']:+d}, degree {ballot['degree']:+d} ({counted}), "
        f"stability {ballot['stability']:+d}, wh {ballot['wh']:+d}, "
        f"factor {ballot['factor']:+d}, margin {ballot['margin']:+d}"
    )


def mechanism(rows: list[dict]) -> dict:
    """Count the cross-cut pattern the losses actually used."""
    counts = {
        "questions": len(rows),
        "pairs": 0,
        "below_cut_entrant": 0,
        "in_window_entrant": 0,
        "scene_tie_factor_only": 0,
        "entrant_wins_pair": 0,
        "entrant_loses_or_ties_pair": 0,
        "kinds": {},
    }
    for row in rows:
        for pair in row["pairs"]:
            counts["pairs"] += 1
            if pair["below_cut"]:
                counts["below_cut_entrant"] += 1
            else:
                counts["in_window_entrant"] += 1
            kind = pair["kinds"]
            counts["kinds"][kind] = counts["kinds"].get(kind, 0) + 1
            ballot = pair["ballot"]
            if ballot is None:
                continue
            if ballot["margin"] > 0:
                counts["entrant_wins_pair"] += 1
            else:
                counts["entrant_loses_or_ties_pair"] += 1
            nonscene = (ballot["degree"] if ballot["degree_counted"] else 0) + ballot["stability"] + ballot["factor"]
            if (
                ballot["scene"] == 0
                and ballot["factor"] == 1
                and ballot["stability"] == 0
                and ballot["wh"] == 0
                and not ballot["degree_counted"]
                and nonscene == 1
            ):
                counts["scene_tie_factor_only"] += 1
    return counts


def markdown(title: str, rows: list[dict], gains: list[dict]) -> str:
    lines = [f"### {title}", ""]
    lines.append(
        "Signs are from the entrant's side. A positive margin is the margin Kemeny used: "
        "the entrant beat that gold turn on this pair. Degree is silent across the cut "
        "in the shipped rule, so a recorded degree sign was not added to the margin."
    )
    lines.append("")
    lines.append(
        "| question | slice | entered top 5 | source | displaced gold | gold now | ballot (entrant vs gold) |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        if not row["pairs"]:
            lines.append(
                f"| `{row['id']}` | {row['slice']} | {', '.join(row['entered']) or '—'} | — | "
                f"{', '.join(row['gold_left']) or '—'} | — | no entrant/gold pair |"
            )
            continue
        for pair in row["pairs"]:
            edge = pair["kinds"]
            if pair["edges"]:
                edge = edge + ": " + "; ".join(pair["edges"])
            lines.append(
                f"| `{row['id']}` | {row['slice']} | `{pair['entrant']}` | {edge} | "
                f"`{pair['gold']}` | {pair['gold_now']} | {tally_line(pair['ballot'])} |"
            )
    lines.append("")
    counts = mechanism(rows)
    lines.append(
        f"Pairs: {counts['pairs']}. Below-cut entrants: {counts['below_cut_entrant']}. "
        f"In-window entrants: {counts['in_window_entrant']}. "
        f"Entrant wins the pair: {counts['entrant_wins_pair']}. "
        f"Entrant ties or loses the pair: {counts['entrant_loses_or_ties_pair']}. "
        f"Scene tie plus a lone factor ballot: {counts['scene_tie_factor_only']}."
    )
    kind_bits = ", ".join(f"{k} {v}" for k, v in sorted(counts["kinds"].items()))
    lines.append(f"Admitting edge on those pairs: {kind_bits}.")
    if gains:
        lines.append("")
        lines.append("Gains, so the net is the card's net: " + ", ".join(f"`{g['id']}` ({g['slice']})" for g in gains) + ".")
    mismatches = [row["id"] for row in rows if row.get("baseline_mismatch")]
    if mismatches:
        lines.append("")
        lines.append("Internal no-factor order disagreed with PR 715 on: " + ", ".join(mismatches) + ".")
    lines.append("")
    return "\n".join(lines)


def lme_questions(hurt: set[str], help_: set[str]):
    path = os.path.join("benchmarks", "data", "longmemeval_s_cleaned.json")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    wanted = hurt | help_
    out = []
    for item in data:
        qid = item["question_id"]
        if qid not in wanted:
            continue
        sessions = []
        gold = set()
        for i, sess in enumerate(item["haystack_sessions"]):
            sid = str(item["haystack_session_ids"][i])
            turns = []
            for j, turn in enumerate(sess):
                tid = f"{sid}#{j}"
                turns.append({"id": tid, "role": str(turn["role"]), "content": str(turn["content"])})
                if turn.get("has_answer") is True:
                    gold.add(tid)
            sessions.append({"id": sid, "date": str(item["haystack_dates"][i] or ""), "turns": turns})
        out.append(
            {
                "id": qid,
                "slice": item["question_type"],
                "lost": qid in hurt,
                "gold": gold,
                "payload": {
                    "sessions": sessions,
                    "queries": [{"text": str(item["question"]), "date": str(item.get("question_date") or "")}],
                    "top_k": 10,
                },
            }
        )
    return out


def beam_questions(hurt: set[str], help_: set[str]):
    base = os.path.join("benchmarks", "BEAM", "chats", "100K")
    wanted = hurt | help_
    convs = sorted({qid.split(":")[0] for qid in wanted}, key=lambda x: int(x))
    out = []
    for c_id in convs:
        chat = json.load(open(os.path.join(base, c_id, "chat.json"), encoding="utf-8"))
        qdata = json.load(
            open(os.path.join(base, c_id, "probing_questions", "probing_questions.json"), encoding="utf-8")
        )
        sessions = []
        for bi, batch in enumerate(chat):
            anchor = batch.get("time_anchor") or ""
            if not anchor and batch.get("turns") and batch["turns"][0]:
                anchor = batch["turns"][0][0].get("time_anchor") or ""
            for sub_idx, sub in enumerate(batch.get("turns") or []):
                if not isinstance(sub, list) or not sub:
                    continue
                sid = f"conv{c_id}_b{bi}_s{sub_idx}"
                turns = []
                for turn in sub:
                    turns.append(
                        {
                            "id": str(turn["id"]),
                            "role": turn.get("role") or "user",
                            "content": turn.get("content") or "",
                        }
                    )
                sessions.append(
                    {
                        "id": sid,
                        "date": (sub[0].get("time_anchor") if sub else None) or anchor or "",
                        "timestamp": 0,
                        "turns": turns,
                    }
                )
        entries = []
        for cat, qlist in qdata.items():
            if not isinstance(qlist, list):
                continue
            for question in qlist:
                gold = list(dict.fromkeys(card.flatten_ids(question.get("source_chat_ids"))))
                if not gold:
                    continue
                entries.append((cat, question["question"], gold))
        keep = []
        for qi, (cat, question, gold) in enumerate(entries):
            qid = f"{c_id}:{cat}:{qi}"
            if qid in wanted:
                keep.append((qi, qid, cat, question, gold))
        if not keep:
            continue
        payload = {
            "sessions": sessions,
            "queries": [{"text": question, "date": ""} for _, question, _ in entries],
            "top_k": 10,
        }
        for qi, qid, cat, _question, gold in keep:
            out.append(
                {
                    "id": qid,
                    "slice": cat,
                    "lost": qid in hurt,
                    "qi": qi,
                    "gold": set(gold),
                    "payload": payload,
                }
            )
    return out


def run_suite(name: str, questions: list[dict], base_bin: str, traced_bin: str):
    base_worker = card.Worker(base_bin)
    env = os.environ.copy()
    env["VESTIGE_FACTOR_TRACE"] = "1"
    proc = subprocess.Popen(
        [traced_bin, "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=env,
    )
    traced_worker = card.Worker.__new__(card.Worker)
    traced_worker.proc = proc
    losses, gains = [], []
    seen_payload = {}
    try:
        for question in questions:
            key = id(question["payload"])
            if key not in seen_payload:
                seen_payload[key] = (
                    base_worker.rank(question["payload"]),
                    traced_worker.rank(question["payload"]),
                )
            base_resp, traced_resp = seen_payload[key]
            qi = question.get("qi", 0)
            base_ids = [turn["id"] for turn in base_resp["results"][qi]["turns"]]
            result = traced_resp["results"][qi]
            row = explain(question, base_ids, result, question["gold"])
            (losses if question["lost"] else gains).append(row)
            print(f"{name} {question['id']} lost={question['lost']}", file=sys.stderr)
    finally:
        base_worker.close()
        traced_worker.close()
    return losses, gains


def main():
    base_bin = sys.argv[1]
    traced_bin = sys.argv[2]
    out_dir = sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    lme_hurt, lme_help, _, _ = loss_ids("/tmp/cards/base-lme.json", "/tmp/cards/v6run1/lme.json")
    beam_hurt, beam_help, _, _ = loss_ids("/tmp/cards/base-beam.json", "/tmp/cards/v6run1/beam.json")
    lme_rows = lme_questions(lme_hurt, lme_help)
    beam_rows = beam_questions(beam_hurt, beam_help)
    lme_losses, lme_gains = run_suite("lme", lme_rows, base_bin, traced_bin)
    beam_losses, beam_gains = run_suite("beam", beam_rows, base_bin, traced_bin)
    lme_losses.sort(key=lambda row: row["id"])
    beam_losses.sort(key=lambda row: row["id"])
    body = "\n".join(
        [
            "## Factor-graph v6 Any@5 forensics",
            "",
            "Same binary rule as commit `920928c0`. PR 715 lists come from that binary. "
            "The traced lists come from the same ranker with the pair ballots attached. "
            "A turn Any@5 loss is a question whose gold was inside the PR 715 top 5 and "
            "outside the factor-graph top 5. LME has 17 such questions and 4 gains (net −13). "
            "BEAM has 8 such questions and 1 gain (net −7).",
            "",
            markdown("LME turn Any@5 losses", lme_losses, lme_gains),
            markdown("BEAM Any@5 losses", beam_losses, beam_gains),
        ]
    )
    path = os.path.join(out_dir, "forensics.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(body)
    blob = {
        "lme_losses": lme_losses,
        "lme_gains": lme_gains,
        "beam_losses": beam_losses,
        "beam_gains": beam_gains,
    }

    def strip(obj):
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items() if k != "payload"}
        if isinstance(obj, list):
            return [strip(v) for v in obj]
        if isinstance(obj, set):
            return sorted(obj)
        return obj

    with open(os.path.join(out_dir, "forensics.json"), "w", encoding="utf-8") as handle:
        json.dump(strip(blob), handle)
    print(body)


if __name__ == "__main__":
    main()
