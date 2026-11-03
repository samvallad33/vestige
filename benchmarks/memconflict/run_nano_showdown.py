#!/usr/bin/env python3
"""MemConflict Showdown: RAG vs. Vestige.

Evaluates how LLMs (local open weights like Qwen2.5-3B-Instruct via Ollama or
micro-models like gpt-5.4-nano) handle memory conflicts (dynamic updates, factual
contradictions, conditional rules) when paired with:
  Arm A: Naive RAG (Okapi BM25 retrieval over historical interaction units)
  Arm B: Vestige Cognitive Memory (Causal graph retrieval via MCP)

Dataset: MemConflict (arXiv:2605.20926) Data/Step4_4.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

import bm25 as bm25_mod
import judge as judge_mod
from mcp_client import VestigeMCP, VestigeMCPError

try:
    import openai
except ImportError:
    openai = None


def is_ollama_model(model: str) -> bool:
    return (
        model.startswith("qwen")
        or model.startswith("ollama/")
        or ":" in model
        or model in ("llama3.1:8b", "granite4.2:8b-q8_0", "gemma4:26b")
    )


def get_openai_client() -> Any:
    if openai is None:
        raise RuntimeError("openai package not installed")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        env_paths = [
            REPO / "benchmarks" / ".env.openai",
            pathlib.Path("/Users/entity002/Documents/Vestige/work/benchmark-campaigns-20261031/.env.openai"),
        ]
        for ep in env_paths:
            if ep.exists():
                key = ep.read_text().strip()
                break
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env.openai")
    return openai.OpenAI(api_key=key)


def load_instances(path: pathlib.Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def session_units(session: Dict[str, Any]) -> List[str]:
    date = session.get("Date", "")
    dialogue = session.get("Session_Dialogue") or {}
    units: List[str] = []

    def turn_no(k: str) -> int:
        try:
            return int(k.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return 0

    for key in sorted(dialogue.keys(), key=turn_no):
        for msg in dialogue[key] or []:
            if msg.get("role") != "user":
                continue
            text = (msg.get("content") or "").strip()
            if text:
                units.append(f"[{date}] {text}")
    return units


def query_llm(model: str, prompt: str, openai_client: Optional[Any] = None) -> Tuple[str, int, int]:
    if is_ollama_model(model):
        clean_model = model.replace("ollama/", "")
        payload = json.dumps({
            "model": clean_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": 60, "temperature": 0.0}
        }).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data.get("message", {}).get("content", "").strip()
            in_tok = data.get("prompt_eval_count", 0)
            out_tok = data.get("eval_count", 0)
            return content, in_tok, out_tok
    else:
        if openai_client is None:
            openai_client = get_openai_client()
        resp = openai_client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=60,
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
        in_tok = resp.usage.input_tokens if hasattr(resp, "usage") else 0
        out_tok = resp.usage.output_tokens if hasattr(resp, "usage") else 0
        return text.strip(), in_tok, out_tok


def make_prompt(context: str, question: str) -> str:
    return (
        "You are an AI assistant answering a specific question about a user based on their chronological interaction history.\n"
        "Guidelines:\n"
        "1. If facts changed or were updated over time, prioritize the most recent, valid state.\n"
        "2. If there is a direct contradiction, state the contradiction explicitly.\n"
        "3. Provide a direct, concise factual answer.\n\n"
        f"Context Memories:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def main():
    default_model = os.environ.get("BENCHMARK_MODEL", "qwen2.5:3b-instruct")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=default_model, help="Model name (e.g. qwen2.5:3b-instruct or gpt-5.4-nano)")
    ap.add_argument("--instances", type=int, default=1, help="Number of simulated users")
    ap.add_argument("--sessions", type=int, default=10, help="Max sessions per user")
    ap.add_argument("--top-k", type=int, default=5, help="Number of memories retrieved")
    ap.add_argument("--binary", default=str(REPO / "target" / "release" / "vestige-mcp"))
    ap.add_argument("--warmup", type=float, default=2.0)
    args = ap.parse_args()

    model_name = args.model
    use_ollama = is_ollama_model(model_name)
    openai_client = None if use_ollama else get_openai_client()

    if use_ollama:
        input_price = 0.0
        output_price = 0.0
    else:
        input_price = 0.10
        output_price = 0.40

    dataset_path = HERE / "data" / "Step4_4.jsonl"
    if not dataset_path.exists():
        sys.exit(f"Dataset not found at {dataset_path}")

    instances = load_instances(dataset_path, args.instances)
    print("=" * 80)
    print(f"  MEMCONFLICT SHOWDOWN: RAG + {model_name} vs. VESTIGE + {model_name}")
    print(f"  Backend: {'Ollama Local' if use_ollama else 'OpenAI API'} | Pricing: ${input_price:.2f}/1M in, ${output_price:.2f}/1M out")
    print(f"  Instances: {len(instances)} | Max Sessions/User: {args.sessions} | Top-K: {args.top_k}")
    print("=" * 80)

    # Initialize Vestige MCP
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    safe_model = model_name.replace(":", "_").replace("/", "_")
    data_dir = HERE / "results" / f"datadir-{safe_model}-{stamp}"
    mcp = VestigeMCP(args.binary, str(data_dir), warmup_seconds=args.warmup)
    mcp.start()
    mcp.initialize()
    print(f"Vestige MCP initialized. Server: {mcp.server_info}\n")

    results = {"rag": [], "vestige": []}
    tokens = {"rag": {"in": 0, "out": 0}, "vestige": {"in": 0, "out": 0}}

    try:
        for inst_idx, inst in enumerate(instances, 1):
            inst_id = inst.get("ID", f"inst_{inst_idx}")
            scope = f"mc_{inst_id.replace('-', '')[:20]}"
            print(f"--- [User {inst_idx}/{len(instances)}] {inst_id} (scope: {scope}) ---")

            rag_corpus: List[str] = []
            chain = sorted(inst.get("Full_Session_Chain", []), key=lambda s: s.get("Session_ID", 0))
            if args.sessions:
                chain = chain[:args.sessions]

            total_q = 0
            for s in chain:
                units = session_units(s)
                if units:
                    # Ingest into RAG
                    rag_corpus.extend(units)

                    # Ingest into Vestige
                    for start in range(0, len(units), 20):
                        chunk = units[start:start+20]
                        items = [
                            {"content": text, "node_type": "event", "tags": [scope], "forceCreate": True}
                            for text in chunk
                        ]
                        mcp.call_tool(
                            "smart_ingest",
                            {
                                "items": items,
                                "scope": scope,
                                "forceCreate": True,
                                "batchMergePolicy": "force_create",
                                "outputProfile": "compact"
                            },
                            timeout=300.0,
                        )

                questions = s.get("Session_Questions") or []
                for q in questions:
                    total_q += 1
                    qtext = q.get("question", "")
                    gold_ans = q.get("answer", "")
                    ctype = q.get("conflict_type", "unknown")
                    qid = q.get("question_id", f"Q_{total_q}")

                    print(f"\n[Q{total_q:02d}] ({ctype}) {qtext}")
                    print(f"  Gold Answer: {gold_ans}")

                    # --- ARM A: RAG (BM25) ---
                    bm25_idx = bm25_mod.BM25(rag_corpus)
                    rag_hits = bm25_idx.top_k(qtext, args.top_k)
                    rag_context = "\n".join(rag_corpus[i] for i, _ in rag_hits)
                    rag_prompt = make_prompt(rag_context, qtext)
                    rag_ans, rag_in, rag_out = query_llm(model_name, rag_prompt, openai_client)
                    tokens["rag"]["in"] += rag_in
                    tokens["rag"]["out"] += rag_out

                    rag_metrics = judge_mod.score_question(q, rag_ans)
                    results["rag"].append({
                        "question_id": qid,
                        "conflict_type": ctype,
                        "gold": gold_ans,
                        "pred": rag_ans,
                        "metrics": rag_metrics,
                    })
                    print(f"  [RAG + {model_name}]     : {rag_ans[:90]}... | AA: {rag_metrics['answer_accuracy']:.2f}")

                    # --- ARM B: VESTIGE ---
                    try:
                        recall_res = mcp.call_tool(
                            "recall",
                            {"handle": scope, "limit": args.top_k, "detail_level": "full"},
                            timeout=120.0,
                        )
                    except Exception:
                        recall_res = mcp.call_tool(
                            "recall",
                            {"query": qtext, "mode": "lookup", "limit": args.top_k, "scope": scope},
                            timeout=120.0,
                        )
                    v_memories = []
                    for k in ("results", "memories", "nodes", "matches"):
                        for item in recall_res.get(k) or []:
                            if isinstance(item, dict):
                                txt = item.get("content") or item.get("preview") or item.get("text")
                                if txt:
                                    v_memories.append(str(txt))
                            elif isinstance(item, str):
                                v_memories.append(item)
                    vestige_context = "\n".join(v_memories[:args.top_k])
                    vestige_prompt = make_prompt(vestige_context, qtext)
                    v_ans, v_in, v_out = query_llm(model_name, vestige_prompt, openai_client)
                    tokens["vestige"]["in"] += v_in
                    tokens["vestige"]["out"] += v_out

                    v_metrics = judge_mod.score_question(q, v_ans)
                    results["vestige"].append({
                        "question_id": qid,
                        "conflict_type": ctype,
                        "gold": gold_ans,
                        "pred": v_ans,
                        "metrics": v_metrics,
                    })
                    print(f"  [Vestige + {model_name}] : {v_ans[:90]}... | AA: {v_metrics['answer_accuracy']:.2f}")

            print(f"\nCompleted User {inst_idx}: {total_q} questions evaluated.")

    finally:
        mcp.close()

    # Summarize Scoreboard
    def summarize(arm_results):
        n = len(arm_results)
        if n == 0:
            return {"aa": 0.0, "dyn_aa": 0.0, "stat_aa": 0.0, "cond_aa": 0.0}
        aa_list = [r["metrics"]["answer_accuracy"] for r in arm_results]
        dyn = [r["metrics"]["answer_accuracy"] for r in arm_results if r["conflict_type"] == "dynamic_conflict"]
        stat = [r["metrics"]["answer_accuracy"] for r in arm_results if r["conflict_type"] == "static_conflict"]
        cond = [r["metrics"]["answer_accuracy"] for r in arm_results if r["conflict_type"] == "conditional_conflict"]
        return {
            "n": n,
            "aa": sum(aa_list) / len(aa_list),
            "dyn_aa": sum(dyn) / len(dyn) if dyn else 0.0,
            "stat_aa": sum(stat) / len(stat) if stat else 0.0,
            "cond_aa": sum(cond) / len(cond) if cond else 0.0,
        }

    sum_rag = summarize(results["rag"])
    sum_vestige = summarize(results["vestige"])

    cost_rag = (tokens["rag"]["in"] * input_price / 1e6) + (tokens["rag"]["out"] * output_price / 1e6)
    cost_vestige = (tokens["vestige"]["in"] * input_price / 1e6) + (tokens["vestige"]["out"] * output_price / 1e6)

    print("\n" + "=" * 80)
    print(f"  MEMCONFLICT SHOWDOWN FINAL RESULTS: RAG vs. VESTIGE (BOTH RUNNING {model_name})")
    print("=" * 80)
    print(f"Total Questions Evaluated: {sum_rag['n']}")
    print(f"Arm A: Naive RAG + {model_name}   | AA: {sum_rag['aa']*100:5.1f}% | Dyn: {sum_rag['dyn_aa']*100:5.1f}% | Stat: {sum_rag['stat_aa']*100:5.1f}% | Cost: ${cost_rag:.5f}")
    print(f"Arm B: Vestige   + {model_name}   | AA: {sum_vestige['aa']*100:5.1f}% | Dyn: {sum_vestige['dyn_aa']*100:5.1f}% | Stat: {sum_vestige['stat_aa']*100:5.1f}% | Cost: ${cost_vestige:.5f}")
    print("-" * 80)
    print(f"Tokens consumed: RAG: {tokens['rag']['in']} in / {tokens['rag']['out']} out | Vestige: {tokens['vestige']['in']} in / {tokens['vestige']['out']} out")
    print("=" * 80)

    # Save output json
    out_file = HERE / "results" / f"showdown_{safe_model}_{stamp}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        json.dump({
            "model": model_name,
            "backend": "ollama" if use_ollama else "openai",
            "summary": {"rag": sum_rag, "vestige": sum_vestige},
            "costs": {"rag": cost_rag, "vestige": cost_vestige},
            "tokens": tokens,
            "results": results,
        }, f, indent=2)
    print(f"Detailed run results saved to {out_file}")


if __name__ == "__main__":
    main()
