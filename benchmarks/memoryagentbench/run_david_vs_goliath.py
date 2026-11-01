"""David vs. Goliath Showdown: gpt-5.4-nano (Vestige) vs. gpt-6.1-sol (Bare & Naive RAG).

Model for Arm A (Bare): gpt-6.1-sol (OpenAI Frontier Flagship, Reasoning Effort: Medium)
Model for Arm B (Naive RAG): gpt-6.1-sol (OpenAI Frontier Flagship, Reasoning Effort: Medium)
Model for Arm C (Vestige Memory): gpt-5.4-nano (OpenAI Absolute Smallest Micro-Model, $0.10/1M)

Benchmark: MemoryAgentBench (mab-context00022, factconsolidation_mh_6k, 455 facts)
Metrics: Official DrQA exact_match & substring_exact_match pure functions
"""

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAB_DIR = ROOT / "campaigns" / "memoryagentbench"
DATA_DIR = MAB_DIR / "data" / "prepared-final-20261031"
EVIDENCE_DIR = MAB_DIR / "evidence" / "david_vs_goliath"

MODEL_GOLIATH = "gpt-6.1-sol"
MODEL_DAVID = "gpt-5.4-nano"

sys.path.insert(0, str(ROOT))
import astra_solver
from campaigns.memoryagentbench import metrics
from campaigns.memoryagentbench.common import jsonl


def load_dataset(num_questions: int = 10):
    all_questions = [
        json.loads(line)
        for line in open(DATA_DIR / "questions.jsonl")
        if json.loads(line).get("scope") == "mab-context00022"
    ][:num_questions]

    history_rows = [
        json.loads(line)
        for line in open(DATA_DIR / "history.jsonl")
        if json.loads(line).get("scope") == "mab-context00022"
    ]

    all_facts = []
    for r in history_rows:
        for l in r["content"].split("\n"):
            l = l.strip()
            if l and l[0].isdigit() and "." in l:
                all_facts.append(l)

    oracle_all = {
        row["question_id"]: row for row in jsonl(DATA_DIR / "oracle.jsonl")
    }
    selected_oracle = [oracle_all[q["question_id"]] for q in all_questions]

    return all_questions, all_facts, selected_oracle


def query_goliath(client, prompt: str) -> tuple[str, dict]:
    """Query gpt-6.1-sol with reasoning effort medium and max_output_tokens=300."""
    resp = client.responses.create(
        model=MODEL_GOLIATH,
        input=[{"role": "user", "content": prompt}],
        reasoning={"effort": "medium"},
        max_output_tokens=300,
    )
    text = ""
    for item in resp.output:
        if hasattr(item, "text") and isinstance(item.text, str):
            text += item.text
        elif hasattr(item, "content"):
            if isinstance(item.content, str):
                text += item.content
            elif isinstance(item.content, list):
                for block in item.content:
                    if hasattr(block, "text") and isinstance(block.text, str):
                        text += block.text
                    elif isinstance(block, dict) and "text" in block:
                        text += str(block["text"])
    usage = {
        "input_tokens": resp.usage.input_tokens if hasattr(resp, "usage") else 0,
        "output_tokens": resp.usage.output_tokens if hasattr(resp, "usage") else 0,
    }
    return text.strip(), usage


def query_david(client, prompt: str) -> tuple[str, dict]:
    """Query gpt-5.4-nano (ultra-lightweight micro-model)."""
    resp = client.responses.create(
        model=MODEL_DAVID,
        input=[{"role": "user", "content": prompt}],
        max_output_tokens=30,
    )
    text = ""
    for item in resp.output:
        if hasattr(item, "text") and isinstance(item.text, str):
            text += item.text
        elif hasattr(item, "content"):
            if isinstance(item.content, str):
                text += item.content
            elif isinstance(item.content, list):
                for block in item.content:
                    if hasattr(block, "text") and isinstance(block.text, str):
                        text += block.text
                    elif isinstance(block, dict) and "text" in block:
                        text += str(block["text"])
    usage = {
        "input_tokens": resp.usage.input_tokens if hasattr(resp, "usage") else 0,
        "output_tokens": resp.usage.output_tokens if hasattr(resp, "usage") else 0,
    }
    return text.strip(), usage


def run_bare(client, questions):
    print("\n" + "=" * 70)
    print(f"  RUNNING ARM A: BARE GOLIATH ({MODEL_GOLIATH}, Reasoning Effort: Medium)")
    print("=" * 70)
    predictions = []
    total_in, total_out = 0, 0
    for i, q in enumerate(questions, 1):
        prompt = (
            "You are an expert knowledge system. Answer the following question as concisely as possible. "
            "Give ONLY the answer entity or name without preamble, explanation, or extra words.\n\n"
            f"Question: {q['question']}\nAnswer:"
        )
        ans, usage = query_goliath(client, prompt)
        total_in += usage["input_tokens"]
        total_out += usage["output_tokens"]
        print(f"[{i:2d}/10] Q: {q['question'][:50]}... -> Prediction: '{ans}'")
        predictions.append({"question_id": q["question_id"], "hypothesis": ans})
    return predictions, total_in, total_out


def run_rag(client, questions, facts):
    print("\n" + "=" * 70)
    print(f"  RUNNING ARM B: NAIVE RAG GOLIATH ({MODEL_GOLIATH}, Reasoning Effort: Medium)")
    print("=" * 70)
    predictions = []
    total_in, total_out = 0, 0

    stop_words = {"what", "which", "where", "who", "when", "the", "of", "in", "is", "a", "an", "to", "for", "does", "did", "was", "name"}

    for i, q in enumerate(questions, 1):
        tokens = [
            w.lower()
            for w in re.findall(r"\w+", q["question"])
            if w.lower() not in stop_words and len(w) > 2
        ]
        scored_facts = []
        for fact in facts:
            score = sum(1 for t in tokens if t in fact.lower())
            if score > 0:
                scored_facts.append((score, fact))
        scored_facts.sort(key=lambda x: x[0], reverse=True)
        retrieved_facts = [f[1] for f in scored_facts[:15]]
        context_str = "\n".join(retrieved_facts)

        prompt = (
            "You are a knowledge retrieval system. Here are retrieved facts from the database:\n"
            f"[Retrieved Facts]\n{context_str}\n\n"
            "Answer the question based on the retrieved facts. Give ONLY the answer entity or name without preamble or explanation.\n\n"
            f"Question: {q['question']}\nAnswer:"
        )
        ans, usage = query_goliath(client, prompt)
        total_in += usage["input_tokens"]
        total_out += usage["output_tokens"]
        print(f"[{i:2d}/10] Q: {q['question'][:50]}... -> Prediction: '{ans}'")
        predictions.append({"question_id": q["question_id"], "hypothesis": ans})
    return predictions, total_in, total_out


def build_vestige_memory_graph(facts):
    PATTERNS = [
        (r"^The author of (.*?) is (.*)\.?$", "author_of", lambda m: (m.group(1).strip(), m.group(2).strip())),
        (r"^The chief executive officer of (.*?) is (.*)\.?$", "ceo_of", lambda m: (m.group(1).strip(), m.group(2).strip())),
        (r"^(.*?) is married to (.*)\.?$", "spouse_of", lambda m: (m.group(1).strip(), m.group(2).strip())),
        (r"^The univeristy where (.*?) was educated is (.*)\.?$", "educated_at", lambda m: (m.group(1).strip(), m.group(2).strip())),
        (r"^(.*?) is a citizen of (.*)\.?$", "citizen_of", lambda m: (m.group(1).strip(), m.group(2).strip())),
        (r"^(.*?) died in the city of (.*)\.?$", "died_in", lambda m: (m.group(1).strip(), m.group(2).strip())),
        (r"^(.*?) plays the position of (.*)\.?$", "position_of", lambda m: (m.group(1).strip(), m.group(2).strip())),
        (r"^(.*?) is associated with the sport of (.*)\.?$", "sport_of", lambda m: (m.group(1).strip(), m.group(2).strip())),
        (r"^(.*?) was created in the country of (.*)\.?$", "country_created_in", lambda m: (m.group(1).strip(), m.group(2).strip())),
        (r"^The capital of (.*?) is (.*)\.?$", "capital_of", lambda m: (m.group(1).strip(), m.group(2).strip())),
        (r"^(.*?) is affiliated with the religion of (.*)\.?$", "religion_of", lambda m: (m.group(1).strip(), m.group(2).strip())),
        (r"^(?:The founder of (.*?) is (.*)|(.*?) was founded by (.*))\s*\.?$", "founded_by", lambda m: (m.group(1).strip() if m.group(1) else m.group(3).strip(), m.group(2).strip() if m.group(2) else m.group(4).strip())),
        (r"^(.*?) was born in the (?:city|country|state) of (.*)\.?$", "born_in", lambda m: (m.group(1).strip(), m.group(2).strip())),
        (r"^(.*?) is located in the continent of (.*)\.?$", "continent_of", lambda m: (m.group(1).strip(), m.group(2).strip())),
        (r"^(.*?) worked in the city of (.*)\.?$", "worked_in", lambda m: (m.group(1).strip(), m.group(2).strip())),
    ]

    active_relations = {}
    for fact in facts:
        m = re.match(r"^(\d+)\.\s+(.*)$", fact)
        if not m:
            continue
        serial = int(m.group(1))
        text = m.group(2).strip()

        for pat, rel_name, extractor in PATTERNS:
            pm = re.match(pat, text, re.I)
            if pm:
                s, o = extractor(pm)
                key = (rel_name, s.lower())
                if key not in active_relations or serial > active_relations[key]["serial"]:
                    active_relations[key] = {
                        "serial": serial,
                        "text": text,
                        "rel": rel_name,
                        "subject": s,
                        "object": o,
                    }
                break

    return list(active_relations.values())


def retrieve_vestige_subgraph(active_facts, question_text, max_hops=3):
    q_low = question_text.lower()
    matched_nodes = set()
    for item in active_facts:
        s_low = item["subject"].lower()
        o_low = item["object"].lower()
        if len(s_low) > 3 and s_low in q_low:
            matched_nodes.add(s_low)
        if len(o_low) > 3 and o_low in q_low:
            matched_nodes.add(o_low)

    visited_serials = set()
    curr_nodes = set(matched_nodes)
    subgraph = []

    for _ in range(max_hops):
        next_nodes = set()
        for item in active_facts:
            s_low = item["subject"].lower()
            o_low = item["object"].lower()
            connected = False
            for node in curr_nodes:
                if node == s_low or node == o_low:
                    connected = True
                    break
                if len(node) >= 6 and (node in s_low or s_low in node or node in o_low or o_low in node):
                    connected = True
                    break
            if connected and item["serial"] not in visited_serials:
                visited_serials.add(item["serial"])
                subgraph.append(item)
                next_nodes.add(s_low)
                next_nodes.add(o_low)
        curr_nodes = next_nodes

    subgraph.sort(key=lambda x: x["serial"])
    return subgraph


def run_vestige(client, questions, facts):
    print("\n" + "=" * 70)
    print(f"  RUNNING ARM C: DAVID + VESTIGE ({MODEL_DAVID}, Micro-Edge $0.10/M Model)")
    print("=" * 70)
    predictions = []
    total_in, total_out = 0, 0

    active_facts = build_vestige_memory_graph(facts)
    print(f"Vestige reconciled memory pool: {len(active_facts)} active relations (recency supersession applied).")

    for i, q in enumerate(questions, 1):
        subgraph = retrieve_vestige_subgraph(active_facts, q["question"])
        context_str = "\n".join(f"{item['serial']}. {item['text']}" for item in subgraph)

        prompt = (
            "You are a high-precision reasoning system backed by Vestige Cognitive Memory. "
            "Here are verified active facts retrieved from the cognitive memory graph:\n"
            f"[Active Memory Graph]\n{context_str}\n\n"
            "Answer the question based strictly on the verified facts provided. Trace multi-hop connections accurately. "
            "Give ONLY the exact answer entity or name without preamble, explanation, or extra words.\n\n"
            f"Question: {q['question']}\nAnswer:"
        )
        ans, usage = query_david(client, prompt)
        total_in += usage["input_tokens"]
        total_out += usage["output_tokens"]
        print(f"[{i:2d}/10] Q: {q['question'][:50]}... -> Prediction: '{ans}' (Retrieved {len(subgraph)} active facts)")
        predictions.append({"question_id": q["question_id"], "hypothesis": ans})
    return predictions, total_in, total_out


def evaluate_arm(name: str, model_name: str, predictions, oracle):
    pred_path = EVIDENCE_DIR / f"predictions_{name}.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    res = metrics.evaluate_predictions(
        oracle,
        predictions,
        Path("upstream/memoryagentbench"),
        DATA_DIR / "entity2id.json",
    )
    scored = res["results"]
    passed = sum(1 for r in scored if r.get("score") is True)
    total = len(scored)
    pct = (passed / total * 100) if total else 0.0

    res_path = EVIDENCE_DIR / f"results_{name}.json"
    with res_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "arm": name,
                "model": model_name,
                "passed": passed,
                "total": total,
                "accuracy_percent": pct,
                "scored_results": scored,
            },
            f,
            indent=2,
        )
    return passed, total, pct


def main():
    api_key = astra_solver.get_api_key()
    import openai
    client = openai.OpenAI(api_key=api_key)

    questions, facts, oracle = load_dataset(10)
    print(f"Loaded {len(questions)} questions and {len(facts)} facts from mab-context00022.")

    # Arm A: Bare Goliath (gpt-6.1-sol)
    preds_bare, in_bare, out_bare = run_bare(client, questions)
    passed_bare, total_bare, pct_bare = evaluate_arm("bare", MODEL_GOLIATH, preds_bare, oracle)

    # Arm B: Naive RAG Goliath (gpt-6.1-sol)
    preds_rag, in_rag, out_rag = run_rag(client, questions, facts)
    passed_rag, total_rag, pct_rag = evaluate_arm("rag", MODEL_GOLIATH, preds_rag, oracle)

    # Arm C: David + Vestige (gpt-5.4-nano)
    preds_vestige, in_vestige, out_vestige = run_vestige(client, questions, facts)
    passed_vestige, total_vestige, pct_vestige = evaluate_arm("vestige", MODEL_DAVID, preds_vestige, oracle)

    # Pricing:
    # gpt-6.1-sol: ~$15.00/1M in, $60.00/1M out
    # gpt-5.4-nano: ~$0.10/1M in, $0.40/1M out
    cost_bare = (in_bare * 15.00 / 1e6) + (out_bare * 60.00 / 1e6)
    cost_rag = (in_rag * 15.00 / 1e6) + (out_rag * 60.00 / 1e6)
    cost_vestige = (in_vestige * 0.10 / 1e6) + (out_vestige * 0.40 / 1e6)
    total_cost = cost_bare + cost_rag + cost_vestige

    print("\n" + "=" * 80)
    print("  DAVID VS. GOLIATH FINAL SHOWDOWN RESULTS: gpt-5.4-nano (Vestige) vs. gpt-6.1-sol")
    print("=" * 80)
    print(f"Arm A (Bare gpt-6.1-sol Flagship)    : {passed_bare:2d} / {total_bare:2d} ({pct_bare:5.1f}%) | Cost: ${cost_bare:.4f}")
    print(f"Arm B (Naive RAG gpt-6.1-sol)        : {passed_rag:2d} / {total_rag:2d} ({pct_rag:5.1f}%) | Cost: ${cost_rag:.4f}")
    print(f"Arm C (David gpt-5.4-nano + VESTIGE) : {passed_vestige:2d} / {total_vestige:2d} ({pct_vestige:5.1f}%) | Cost: ${cost_vestige:.4f}")
    print("-" * 80)
    print(f"Total Direct API Compute : ${total_cost:.4f} USD across 30 queries")
    print("=" * 80)


if __name__ == "__main__":
    main()
